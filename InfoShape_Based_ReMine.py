# %% Import Necessary Libraries
import numpy as np
import json
import torch
import matplotlib.pyplot as plt

import torchvision
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything

import torch.nn as nn
from torch.nn import functional as F

import math
import logging
logging.getLogger().setLevel(logging.ERROR)

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# %% Set Cpmputing Device and Set Random Seed
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda:0 or cuda:1 for our 2-GPU testbed
num_gpus = 1 if device=='cuda:0' else 0
print(device)

seed = 1
seed_everything(seed, workers=True)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Numpy RNG
np.random.seed(seed)

# %% Parameters and Constants
DATA_SAVE_PATH = "data/"  # Location for saved classification datapoints
ENC_SAVE_PATH = "model/"  # Location for saving encoder weights per training epoch
FIG_SAVE_PATH = "figures/" # Location for saving graphical results

BETA = 1  # Used for the Lagrangian optimization in the loss term for InfoShape, i.e., maximizing I(T(X);L(X)) - BETA * I(T(X);S(X))
N_ENC_OUT_NODES = 10  # Number of nodes in the task-specific encoder (InfoShape)'s output layer 
N_CLASSIFIER_TRAINING_EPOCHS = 10 # Number of epochs for training classifiers
N_INFOSHAPE_EPOCHS = 50 # Number of epochs for training InfoShape

EPS = 1e-6 # Used in computing the gradient of loss when numerically estimating MI
MINE_EPOCHS = 2000 # # Number of ietartions for numerically estimating MI
MINE_BATCH_SIZE = 5000  # batch size for loading dataset into MINE

# %% Dense neural networks that are used as Classifier and as Task-Specific Encoder

class DenseClassifier(nn.Module):
    def __init__(self, in_nodes, hidden_nodes=20):
        super(DenseClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_nodes, hidden_nodes), 
            nn.ReLU(), 
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(device)
        return self.main(x)

    def train_classifier(self, train_loader, epochs=N_CLASSIFIER_TRAINING_EPOCHS):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
        for epoch in tqdm(range(epochs)):
            self.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device).float(), y.to(device).float()
                y_hat = self(x).squeeze()
                loss = F.binary_cross_entropy(y_hat, y, reduction="sum")
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Average loss per sample
            # If avg loss per elements, divide again by size of each sample
            avg_train_loss = train_loss / len(train_loader.dataset)
            print(f'====> Epoch: {epoch} Average loss: {avg_train_loss:.4f}')

    def evaluate(self, test_loader):
        self.eval()
        test_data, test_labels = test_loader.dataset.data, test_loader.dataset.targets
        preds = self(test_data.to(device)).squeeze()
        y_true = test_labels
        y_score = preds.detach().cpu().numpy()

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        test_loss = F.binary_cross_entropy(torch.from_numpy(y_true).float(), torch.from_numpy(y_score).float())

        return fpr, tpr, thresholds, auc, test_loss

class DenseEncoder(nn.Module):
    def __init__(self, in_dim, hidden_nodes=50, out_nodes=N_ENC_OUT_NODES):
        super(DenseEncoder, self).__init__()
        in_nodes = in_dim[0]
        self.out_nodes = out_nodes
        self.main = nn.Sequential(
            # Want 2 layers for more nonlinearity in the encoded data
            nn.Linear(in_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, out_nodes),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.to(device)
        return self.main(x)

# %% Constructing the original (un-encoded noiseless) Synthetic Dataset

class SyntheticDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

# Digit MNIST Dataset
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=True, download=True)

BATCH_SIZE = 100 # Batch size for training classifiers
N_FEATURES = train.data.shape[1]*train.data.shape[2] # Number of features per sample 28x28
GRADIENT_BATCH_SIZE =  train.data.shape[0]/MINE_BATCH_SIZE  # How many gradients get accumulated before update (zero_grad)

X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets
X_test = test.data.reshape(-1, 784).float() / 255.0
y_test = test.targets

# Public label set, public label odd or even digits, private label greater than 4 or not
y_train_pub = np.where(y_train % 2 == 0, np.ones(y_train.shape), np.zeros_like(y_train))
y_train_pri = np.where(y_train > 4, np.ones(y_train.shape), np.zeros_like(y_train))
y_test_pub = np.where(y_test % 2 == 0, np.ones(y_test.shape), np.zeros_like(y_test))
y_test_pri = np.where(y_test > 4, np.ones(y_test.shape), np.zeros_like(y_test))

# og stands for original noiseless dataset
og_train_dataset_pub = SyntheticDataset(X_train, y_train_pub)
og_test_dataset_pub = SyntheticDataset(X_test, y_test_pub)
og_train_dataset_pri = SyntheticDataset(X_train, y_train_pri)
og_test_dataset_pri = SyntheticDataset(X_test, y_test_pri)

og_train_loader_pub = DataLoader(og_train_dataset_pub, batch_size=BATCH_SIZE, shuffle=True)
og_test_loader_pub = DataLoader(og_test_dataset_pub, batch_size=BATCH_SIZE, shuffle=True)
og_train_loader_pri = DataLoader(og_train_dataset_pri, batch_size=BATCH_SIZE, shuffle=True)
og_test_loader_pri = DataLoader(og_test_dataset_pri, batch_size=BATCH_SIZE, shuffle=True)

# %%  Analysis and Visualization of Original Data Points
# USe T_SNE method to convert the high-dimensional samples to lower dimensional samples with close distributions

tsne = TSNE(n_components=2, verbose=1, random_state=seed)
Z_train = tsne.fit_transform(X_train)

# public label
cdict = {0: 'gold', 1: 'indigo'}
plt.figure()
ix1 = np.where(y_train_pub == 0)
ix2 = np.where(y_train_pub == 1)
plt.scatter(Z_train[ix1,0], Z_train[ix1,1], c = cdict[0], s=0.1)
plt.scatter(Z_train[ix2,0], Z_train[ix2,1], c = cdict[1], s=0.1)
plt.savefig( FIG_SAVE_PATH + 'og_dataset_pub.png' )

# private label
plt.figure()
cdict = {0: 'deepskyblue', 1: 'darkorange'}
ix1 = np.where(y_train_pri == 0)
ix2 = np.where(y_train_pri == 1)
plt.scatter(Z_train[ix1,0], Z_train[ix1,1], c = cdict[0], s=0.1)
plt.scatter(Z_train[ix2,0], Z_train[ix2,1], c = cdict[1], s=0.1)
plt.savefig( FIG_SAVE_PATH + 'og_dataset_pri.png' )

# %% Constructing Noisy Dataset (baseline)

# Independently Adding Normal Guassian Noise per element of each sample
MU = 0
VARIANCE = 1
STDDEV = VARIANCE ** 0.5

noise_matrix = np.zeros_like(X_train)
for i in range(X_train.shape[0]):
    noise_matrix[i] = np.random.normal(MU, STDDEV, N_FEATURES)
X_train_noisy = X_train + noise_matrix

noise_matrix = np.zeros_like(X_test)
for i in range(X_test.shape[0]):
    noise_matrix[i] = np.random.normal(MU, STDDEV, N_FEATURES)
X_test_noisy = X_test + noise_matrix

noisy_train_dataset_pub = SyntheticDataset(X_train_noisy, y_train_pub)
noisy_test_dataset_pub = SyntheticDataset(X_test_noisy, y_test_pub)
noisy_train_dataset_pri = SyntheticDataset(X_train_noisy, y_train_pri)
noisy_test_dataset_pri = SyntheticDataset(X_test_noisy, y_test_pri)

noisy_train_loader_pub = DataLoader(noisy_train_dataset_pub, batch_size=BATCH_SIZE, shuffle=True)
noisy_test_loader_pub = DataLoader(noisy_test_dataset_pub, batch_size=BATCH_SIZE, shuffle=True)
noisy_train_loader_pri = DataLoader(noisy_train_dataset_pri, batch_size=BATCH_SIZE, shuffle=True)
noisy_test_loader_pri = DataLoader(noisy_test_dataset_pri, batch_size=BATCH_SIZE, shuffle=True)


# %% Experiments: Training Classifiers on Original and Noisy Datasets

# Public Label

model = DenseClassifier(N_FEATURES).to(device)
model.train_classifier(og_train_loader_pub, epochs=N_CLASSIFIER_TRAINING_EPOCHS) # Original Data
pub_og_fpr, pub_og_tpr, pub_og_thresholds, pub_og_auc, pub_og_test_loss = model.evaluate(og_test_loader_pub)
with open(DATA_SAVE_PATH + 'og_data_roc_auc_pub.json', "w") as f:
    json.dump({"fpr": pub_og_fpr.tolist(), "tpr": pub_og_tpr.tolist(), "thresholds": pub_og_thresholds.tolist()}, f)

model = DenseClassifier(N_FEATURES).to(device)
model.train_classifier(noisy_train_loader_pub, epochs=N_CLASSIFIER_TRAINING_EPOCHS) # Noisy Data
pub_noisy_fpr, pub_noisy_tpr, pub_noisy_thresholds, pub_noisy_auc, pub_noisy_test_loss  = model.evaluate(noisy_test_loader_pub) 
with open(DATA_SAVE_PATH + 'noisy_data_roc_auc_pub.json', "w") as f:
    json.dump({"fpr": pub_noisy_fpr.tolist(), "tpr": pub_noisy_tpr.tolist(), "thresholds": pub_noisy_thresholds.tolist()}, f)

# # Private Label

model = DenseClassifier(N_FEATURES).to(device)
model.train_classifier(og_train_loader_pri, epochs=N_CLASSIFIER_TRAINING_EPOCHS) # Original Data
pri_og_fpr, pri_og_tpr, pri_og_thresholds, pri_og_auc, pri_og_test_loss = model.evaluate(og_test_loader_pri)
with open(DATA_SAVE_PATH + 'og_data_roc_auc_pri.json', "w") as f:
    json.dump({"fpr": pri_og_fpr.tolist(), "tpr": pri_og_tpr.tolist(), "thresholds": pri_og_thresholds.tolist()}, f)

model = DenseClassifier(N_FEATURES).to(device)
model.train_classifier(noisy_train_loader_pri, epochs=N_CLASSIFIER_TRAINING_EPOCHS) # Noisy Data
pri_noisy_fpr, pri_noisy_tpr, pri_noisy_thresholds, pri_noisy_auc, pri_noisy_test_loss  = model.evaluate(noisy_test_loader_pri) 
with open(DATA_SAVE_PATH + 'noisy_data_roc_auc_pri.json', "w") as f:
    json.dump({"fpr": pri_noisy_fpr.tolist(), "tpr": pri_noisy_tpr.tolist(), "thresholds": pri_noisy_thresholds.tolist()}, f)

# %% Mutual Information Estimation Setup: Inspired by MINE and ReMINE papers

class T(nn.Module): # This is the function that its parameters are optimized to be used in estimation of MI
    def __init__(self, enc_out_num_nodes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(enc_out_num_nodes + 1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, z, labels):
        z, labels = z.float().to(device), labels.float().to(device)
        z = z.view(z.size(0), -1).to(device)
        cat = torch.cat((z, labels.unsqueeze(-1)), 1).to(device)
        return self.layers(cat).to(device)    

class EMALoss(torch.autograd.Function): # exponential moving average.
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output): # Gradient
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None

def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach() # The second term is for going from sum to average
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = alpha * t_exp + (1.0 - alpha) * running_mean.item()
    t_log = EMALoss.apply(x, running_mean) # Forward

    return t_log, running_mean

class Mine(nn.Module):
    def __init__(self, stats_network, loss='mine', alpha=0.01, lam=0.1, C=0):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha  # Used for ema during MINE iterations
        # Both lambda and C are a part of the regularization in ReMINE's objective
        self.lam = lam # Lambda
        self.C = C
        self.stats_network = stats_network # Function stat_net

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])] # Permutation of z for marginal distribution

        stats_network_score = self.stats_network(x, z).mean() # The first terms in Remine Estimation
        t_marg = self.stats_network(x, z_marg)
        

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        # Introducing ReMINE regularization here
        return -stats_network_score + second_term + self.lam * (second_term - self.C) ** 2 # Minus sign is because of the minimization
    
class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, loss='mine', **kwargs):
        super().__init__()
        self.energy_loss = kwargs.get('mine')
        self.file_name = kwargs.get('file_name') + ".txt"
        self.kwargs = kwargs
        self.gradient_batch_size = kwargs.get('gradient_batch_size', 1)
        self.train_loader = kwargs.get('train_loader')
        assert self.energy_loss is not None
        assert self.train_loader is not None
        print("energy loss: ", self.energy_loss)
        with open(DATA_SAVE_PATH + self.file_name, 'w') as f:
            pass # clear the file

    def forward(self, x, z):
        if self.on_gpu:
            x = x.to(device)
            z = z.to(device)

        return self.energy_loss(x, z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])

    def training_step(self, batch, batch_idx):
        x, z = batch
        if self.on_gpu:
            x = x.to(device)
            z = z.to(device)

        loss = self.energy_loss(x, z).to(device)
        mi = -loss
        tensorboard_logs = {'loss': loss, 'mi': mi}
        tqdm_dict = {'loss_tqdm': loss, 'mi': mi}
        self.last_mi = mi
        self.logger.experiment.add_scalar(
            f"MI Train",
            self.current_epoch,
            mi
        )
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        
        if batch_idx % self.gradient_batch_size == 0:
            with open(DATA_SAVE_PATH + self.file_name, 'a') as f:
                f.write(str(self.current_epoch)+'\t'+str(mi.tolist())+'\n')
        return {
            **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict
        }
        
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False):
        if batch_idx % self.gradient_batch_size == 0:
            optimizer.step(closure=optimizer_closure)
        else:
            # REFACTOR: Aassumes optimizer closure always non-null
            optimizer_closure()

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int):
        if batch_idx % self.gradient_batch_size == 0:
            optimizer.zero_grad()

    def train_dataloader(self):
        assert self.train_loader is not None
        return self.train_loader
    
# %% Calculate H(L(X))=I(X,L(X))

EXPERIMENT = f"SYNTHETIC DATA REMINE BS=2K C=0 Lambda=0.1 AVG 10"
train_loader_pub_HLx = DataLoader(og_train_dataset_pub, batch_size=MINE_BATCH_SIZE, shuffle=True)

t = T(N_FEATURES).to(device)
mi_estimator = Mine(t, loss='mine').to(device)
func_str = f"I(x;L(x))"
lr = 1e-4

kwargs = {
    'mine': mi_estimator,
    'lr': lr,
    'batch_size': MINE_BATCH_SIZE,
    'alpha': 0.1,
    'func': func_str,
    'train_loader': train_loader_pub_HLx,
    # Determines how many minibatches (MINE iters) of gradients get accumulated before optimizer step gets applied
    # Meant to stabilize the MINE curve for better encoder training performance
    'gradient_batch_size': GRADIENT_BATCH_SIZE,
    'file_name': 'I(X,L(X))'
}

logger = TensorBoardLogger(
    "lightning_logs",
    name=f"{EXPERIMENT} utility BS={MINE_BATCH_SIZE}",
    version=f"{func_str}, BS: {MINE_BATCH_SIZE}"
)

model = MutualInformationEstimator(loss='mine', **kwargs).to(device)

trainer = Trainer(max_epochs=MINE_EPOCHS, logger=logger, gpus=1)
trainer.fit(model)

# %% Calculate H(S(X))=I(X,S(X))

EXPERIMENT = f"SYNTHETIC DATA REMINE BS=2K C=0 Lambda=0.1 AVG 10"
train_loader_pub_HSx = DataLoader(og_train_dataset_pri, batch_size=MINE_BATCH_SIZE, shuffle=True)

t = T(N_FEATURES).to(device)
mi_estimator = Mine(t, loss='mine').to(device)
func_str = f"I(x;S(x))"
lr = 1e-4

kwargs = {
    'mine': mi_estimator,
    'lr': lr,
    'batch_size': MINE_BATCH_SIZE,
    'alpha': 0.1,
    'func': func_str,
    'train_loader': train_loader_pub_HSx,
    # Determines how many minibatches (MINE iters) of gradients get accumulated before optimizer step gets applied
    # Meant to stabilize the MINE curve for better encoder training performance
    'gradient_batch_size': GRADIENT_BATCH_SIZE,
    'file_name': 'I(X,S(X))'
}

logger = TensorBoardLogger(
    "lightning_logs",
    name=f"{EXPERIMENT} utility BS={MINE_BATCH_SIZE}",
    version=f"{func_str}, BS={MINE_BATCH_SIZE}"
)

model = MutualInformationEstimator(loss='mine', **kwargs).to(device)

trainer = Trainer(max_epochs=MINE_EPOCHS, logger=logger, gpus=1)
trainer.fit(model)

# %% Classification on Encoded Data (Untrained Encoder)
# Using untrained encoder

# Public Labels

enc = DenseEncoder((N_FEATURES,), out_nodes=N_ENC_OUT_NODES).to(device)
train_transform_pub = enc(og_train_dataset_pub.data.float()).detach()
test_transform_pub = enc(og_test_dataset_pub.data.float()).detach()
train_data_transform_pub = SyntheticDataset(
    train_transform_pub,
    og_train_dataset_pub.targets
)
test_data_transform_pub = SyntheticDataset(
    test_transform_pub,
    og_test_dataset_pub.targets
)

rand_train_loader_transform_pub = DataLoader(train_data_transform_pub, batch_size=BATCH_SIZE, shuffle=True)
rand_test_loader_transform_pub = DataLoader(test_data_transform_pub, batch_size=BATCH_SIZE, shuffle=True)

model_enctrans = DenseClassifier(enc.out_nodes).to(device)

model_enctrans.train_classifier(rand_train_loader_transform_pub, epochs=N_CLASSIFIER_TRAINING_EPOCHS)
pub_rand_fpr, pub_rand_tpr, pub_rand_thresholds, pub_rand_auc, pub_rand_test_loss = model_enctrans.evaluate(rand_test_loader_transform_pub)

with open(DATA_SAVE_PATH + 'randomized_data_roc_auc_pub.json', "w") as f:
   json.dump({"fpr": pub_rand_fpr.tolist(), "tpr": pub_rand_tpr.tolist(), "thresholds": pub_rand_thresholds.tolist()}, f)

# Private Labels

enc = DenseEncoder((N_FEATURES,), out_nodes=N_ENC_OUT_NODES).to(device)
train_transform_pri = enc(og_train_dataset_pri.data.float()).detach()
test_transform_pri = enc(og_test_dataset_pri.data.float()).detach()
train_data_transform_pri = SyntheticDataset(
    train_transform_pri,
    og_train_dataset_pri.targets
)
test_data_transform_pri = SyntheticDataset(
    test_transform_pri,
    og_test_dataset_pri.targets
)

rand_train_loader_transform_pri = DataLoader(train_data_transform_pri, batch_size=BATCH_SIZE, shuffle=True)
rand_test_loader_transform_pri = DataLoader(test_data_transform_pri, batch_size=BATCH_SIZE, shuffle=True)

model_enctrans = DenseClassifier(enc.out_nodes).to(device)

model_enctrans.train_classifier(rand_train_loader_transform_pri, epochs=N_CLASSIFIER_TRAINING_EPOCHS)
pri_rand_fpr, pri_rand_tpr, pri_rand_thresholds, pri_rand_auc, pri_rand_test_loss = model_enctrans.evaluate(rand_test_loader_transform_pri)

with open(DATA_SAVE_PATH + 'randomized_data_roc_auc_pri.json', "w") as f:
   json.dump({"fpr": pri_rand_fpr.tolist(), "tpr": pri_rand_tpr.tolist(), "thresholds": pri_rand_thresholds.tolist()}, f)

# %% [INFOSHAPE SETUP] Dual Optimization Procedure

class DualOptimizationDenseEncoder(nn.Module):
    def __init__(self, data_loader, mine_epochs_privacy, mine_epochs_utility, enc_out_nodes=N_ENC_OUT_NODES, beta=BETA, enc_shape=N_FEATURES, private_labels=None):
        super().__init__()
        self.encoder = DenseEncoder((enc_shape,), out_nodes=enc_out_nodes).to(device)
        self.data_loader = data_loader
        self.private_labels = torch.from_numpy(private_labels)  # Fail fast with None value if misconfigured
        self.mine_epochs_privacy = mine_epochs_privacy
        self.mine_epochs_utility = mine_epochs_utility
        self.beta = beta

    def get_MINE(self, transformed_data_loader, enc_out_num_nodes, mine_epochs, train_epoch, filename, K=MINE_BATCH_SIZE, gradient_batch_size=1, func_str=None):
        stats_network = T(enc_out_num_nodes).to(device)
        mi_estimator = Mine(stats_network, loss='mine').to(device)
        func_str = f"training epoch={train_epoch}: f(x)=DenseEnc(x) {enc_out_num_nodes} nodes" if not func_str else func_str

        kwargs = {
            'mine': mi_estimator,
            'lr': 1e-4,
            'batch_size': K,
            'alpha': 0.1,  # Used as the ema weight in MINE
            'func': func_str,
            'train_loader': transformed_data_loader,
            # Determines how many minibatches (MINE iters) of gradients get accumulated before optimizer step gets applied
            # Meant to stabilize the MINE curve for [hopefully] better encoder training performance
            'gradient_batch_size': gradient_batch_size,
            'file_name': filename
        }
        
        logger = TensorBoardLogger(
            "lightning_logs",
            name=f"{EXPERIMENT} BS={K}",
            version=f"{func_str}, BS={K}"
        )

        model = MutualInformationEstimator(loss='mine', **kwargs).to(device)
        return model, logger

    def forward(self, epoch, num_batches_final_MI, include_privacy=True, include_utility=True, K=MINE_BATCH_SIZE, gradient_batch_size=1):
        # Get encoder transformed data
        transformedsamples = self.encoder(self.data_loader.dataset.data.float())

        labels_public = self.data_loader.dataset.targets
        labels_private = self.private_labels

        z_train_utility_detached = SyntheticDataset(transformedsamples.detach(), labels_public)
        z_train_privacy_detached = SyntheticDataset(transformedsamples.detach(), labels_private)
        z_train_loader_utility_detached = DataLoader(z_train_utility_detached, K, shuffle=True)
        z_train_loader_privacy_detached = DataLoader(z_train_privacy_detached, K, shuffle=True)

        # Get MINE model (sitting in Pytorch lightning module)
        model_MINE_utility, logger_utility = self.get_MINE(
            z_train_loader_utility_detached, self.encoder.out_nodes, self.mine_epochs_utility, epoch, 'I(T(X),L(X)) epoch %i'%epoch, K=K, gradient_batch_size=gradient_batch_size)
        model_MINE_privacy, logger_privacy = self.get_MINE(
            z_train_loader_privacy_detached, self.encoder.out_nodes, self.mine_epochs_privacy, epoch, 'I(T(X),S(X)) epoch %i'%epoch, K=K, gradient_batch_size=gradient_batch_size)

        # Optimize MINE estimate, "train" MINE
        last_mi_utility = last_mi_privacy = 0
        if include_utility:
            trainer_utility = Trainer(max_epochs=self.mine_epochs_utility, logger=logger_utility, gpus=1)
            trainer_utility.fit(model_MINE_utility)

            ## -------- Calculate I(T(x); L(x)) estimate after MINE training ---------- ##
            # **IMPORTANT**: Use the non-detached og transformed samples so that gradients are retained
            z_train_utility = SyntheticDataset(transformedsamples, labels_public)
            z_train_loader_utility = DataLoader(z_train_utility, K, shuffle=True)
            model_MINE_utility.energy_loss.to(device)
            sum_MI_utility = 0

            # Average MI across num_batches_final_MI batches to lower variance
            # Batches are K random samples from the dataset after all
            assert num_batches_final_MI < len(z_train_loader_utility.dataset) / K
            utility_it = iter(z_train_loader_utility)
            for i in range(num_batches_final_MI):
                Tx, Lx = next(utility_it)
                Tx.to(device)
                Lx.to(device)
                sum_MI_utility += model_MINE_utility.energy_loss(Tx, Lx)
                
            # MINE loss = -1 * MI estimate since we are maximizing using gradient descent still
            last_mi_utility = -1 * sum_MI_utility / num_batches_final_MI

        if include_privacy:
            trainer = Trainer(max_epochs=self.mine_epochs_privacy, logger=logger_privacy, gpus=1)
            trainer.fit(model_MINE_privacy)

            ## -------- Calculate I(T(x); S(x)) estimate after MINE training ---------- ##
            z_train_privacy = SyntheticDataset(transformedsamples, labels_private)
            z_train_loader_privacy = DataLoader(z_train_privacy, K, shuffle=True)
            model_MINE_privacy.energy_loss.to(device)

            assert num_batches_final_MI < len(z_train_loader_privacy.dataset) / K
            sum_MI_privacy = 0
            privacy_it = iter(z_train_loader_privacy)
            for i in range(num_batches_final_MI):
                Tx, Sx = next(privacy_it)
                Tx.to(device)
                Sx.to(device)
                sum_MI_privacy += model_MINE_privacy.energy_loss(Tx, Sx)
                
            last_mi_privacy = -1 * sum_MI_privacy / num_batches_final_MI
        return last_mi_utility, last_mi_privacy

    def train_encoder(
        self, 
        num_enc_epochs=10, 
        num_batches_final_MI=100, 
        save_enc_weights=False, 
        include_privacy=True, 
        include_utility=True,
        K=MINE_BATCH_SIZE,
        gradient_batch_size=1,
        enc_save_path=ENC_SAVE_PATH,
    ):
        # Encoder's training params
        learning_rate = 1e-3
        encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=learning_rate,
        )
        self.encoder.train()

        for epoch in range(num_enc_epochs):
            mi_utility, mi_privacy = self.forward( 
                epoch, num_batches_final_MI, include_privacy=include_privacy, include_utility=include_utility, K=K, gradient_batch_size=gradient_batch_size
            )
            encoder_optimizer.zero_grad()            
            loss = -mi_utility + self.beta * mi_privacy
            loss.backward()
            encoder_optimizer.step()

            if save_enc_weights:
                # Don't save the state dict since that doesn't include the model parameters + their gradients
                # Options were to save entire model or optimizer's state dict:
                # https://discuss.pytorch.org/t/how-to-save-the-requires-grad-state-of-the-weights/52906/6
                print(f"Saving weights to {enc_save_path}")
                torch.save(self.encoder, enc_save_path + f"{EXPERIMENT} epoch={epoch}.pt")
                torch.save(encoder_optimizer.state_dict(), enc_save_path + f"[optimizer] {EXPERIMENT} epoch={epoch}.pt")

            print(f'====> Epoch: {epoch} Utility MI I(T(x); L(x)): {mi_utility:.8f}')
            print(f'====> Epoch: {epoch} Privacy MI I(T(x); S(x)): {mi_privacy:.8f}')
            print(f'====> Epoch: {epoch} Loss: {loss:.8f}')

# %% [INFOSHAPE] Training Procedure

EXPERIMENT = f"[DEBUG][SYNTHETIC DATA | PUBLIC PRIVATE LABELS | DUAL ENC TRAIN] REMINE BS=2K C=0 λ=0.1 MIb=10 ENC_EPOCHS={N_INFOSHAPE_EPOCHS}"

dualopt_model = DualOptimizationDenseEncoder(
    og_train_loader_pub,
    mine_epochs_privacy=MINE_EPOCHS,
    mine_epochs_utility=MINE_EPOCHS,
    enc_out_nodes=N_ENC_OUT_NODES,
    private_labels=y_train_pri
).to(device)

dualopt_model.train_encoder(
    num_enc_epochs=N_INFOSHAPE_EPOCHS,
    num_batches_final_MI=3,
    include_privacy=True,
    include_utility=True,
    K=MINE_BATCH_SIZE,
    gradient_batch_size=GRADIENT_BATCH_SIZE,
    save_enc_weights=True,
    enc_save_path=ENC_SAVE_PATH
)

# %% Training Classifiers on Infoshape encoded data
enc_trained = torch.load(ENC_SAVE_PATH + f"{EXPERIMENT} epoch={N_INFOSHAPE_EPOCHS-1}.pt").to(device)  # epochs are 0 indexed

# Public Labels

train_transform_pub_trainedenc = enc_trained(og_train_dataset_pub.data.float()).detach()
test_transform_pub_trainedenc = enc_trained(og_test_dataset_pub.data.float()).detach()
train_data_transform_pub_trainedenc = SyntheticDataset(
    train_transform_pub_trainedenc,
    og_train_dataset_pub.targets
)
test_data_transform_pub_trainedenc = SyntheticDataset(
    test_transform_pub_trainedenc,
    og_test_dataset_pub.targets
)

train_loader_transform = DataLoader(train_data_transform_pub_trainedenc, batch_size=BATCH_SIZE, shuffle=True)
test_loader_transform = DataLoader(test_data_transform_pub_trainedenc, batch_size=BATCH_SIZE, shuffle=True)

model_enctrans_ = DenseClassifier(enc_trained.out_nodes).to(device)
model_enctrans_.train_classifier(train_loader_transform, epochs=N_CLASSIFIER_TRAINING_EPOCHS)

pub_InfoShape_fpr, pub_InfoShape_tpr, pub_InfoShape_thresholds, pub_InfoShape_auc, pub_InfoShape_test_loss = model_enctrans_.evaluate(test_loader_transform)

with open(DATA_SAVE_PATH + 'InfoShape_data_roc_auc_pub.json', "w") as f:
   json.dump({"fpr": pub_InfoShape_fpr.tolist(), "tpr": pub_InfoShape_tpr.tolist(), "thresholds": pub_InfoShape_thresholds.tolist()}, f)

# Private Labels

train_transform_pri_trainedenc = enc_trained(og_train_dataset_pri.data.float()).detach()
test_transform_pri_trainedenc = enc_trained(og_test_dataset_pri.data.float()).detach()
train_data_transform_pri_trainedenc = SyntheticDataset(
    train_transform_pri_trainedenc,
    og_train_dataset_pri.targets
)
test_data_transform_pri_trainedenc = SyntheticDataset(
    test_transform_pri_trainedenc,
    og_test_dataset_pri.targets
)

train_loader_transform = DataLoader(train_data_transform_pri_trainedenc, batch_size=BATCH_SIZE, shuffle=True)
test_loader_transform = DataLoader(test_data_transform_pri_trainedenc, batch_size=BATCH_SIZE, shuffle=True)

model_enctrans_ = DenseClassifier(enc_trained.out_nodes).to(device)
model_enctrans_.train_classifier(train_loader_transform, epochs=N_CLASSIFIER_TRAINING_EPOCHS)

pri_InfoShape_fpr, pri_InfoShape_tpr, pri_InfoShape_thresholds, pri_InfoShape_auc, pri_InfoShape_test_loss = model_enctrans_.evaluate(test_loader_transform)

with open(DATA_SAVE_PATH + 'InfoShape_data_roc_auc_pri.json', "w") as f:
   json.dump({"fpr": pri_InfoShape_fpr.tolist(), "tpr": pri_InfoShape_tpr.tolist(), "thresholds": pri_InfoShape_thresholds.tolist()}, f)

# # %% Printing ROC curves and AUC values

plt.figure()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate' )
plt.plot(pub_og_fpr, pub_og_tpr, marker='.', label='Original Data')
plt.plot(pub_noisy_fpr, pub_noisy_tpr, marker='.', label='Noisy Data')
plt.plot(pub_rand_fpr, pub_rand_tpr, marker='.', label='Randomly Encoded Data')
plt.plot(pub_InfoShape_fpr, pub_InfoShape_tpr, marker='.', label='InfoShape Encoded Data')
plt.legend()
plt.savefig(FIG_SAVE_PATH + 'public_label_ROC.png' )

plt.figure()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate' )
plt.plot(pri_og_fpr, pri_og_tpr, marker='.', label='Original Data')
plt.plot(pri_noisy_fpr, pri_noisy_tpr, marker='.', label='Noisy Data')
plt.plot(pri_rand_fpr, pri_rand_tpr, marker='.', label='Randomly Encoded Data')
plt.plot(pri_InfoShape_fpr, pri_InfoShape_tpr, marker='.', label='InfoShape Encoded Data')
plt.legend()
plt.savefig(FIG_SAVE_PATH + 'private_label_ROC.png' )

print("Public Label AUC: ", pub_og_auc, pub_noisy_auc, pub_rand_auc, pub_InfoShape_auc)
print("Private Label AUC: ", pri_og_auc, pri_noisy_auc, pri_rand_auc, pri_InfoShape_auc)

# # %% Printing MI Estimations at carious epochs

itr_vec = range(MINE_EPOCHS)

pub_MI_epoch_first = []
pub_MI_epoch_middle = []
pub_MI_epoch_last = []
pub_MI_og = []

for line in open(DATA_SAVE_PATH+'I(T(X),L(X)) epoch 0.txt', "r"):
    itr, est = line.split()
    pub_MI_epoch_first.append(float(est))
for line in open(DATA_SAVE_PATH+'I(T(X),L(X)) epoch 14.txt', "r"):
    itr, est = line.split()
    pub_MI_epoch_middle.append(float(est))
for line in open(DATA_SAVE_PATH+'I(T(X),L(X)) epoch 49.txt', "r"):
    itr, est = line.split()
    pub_MI_epoch_last.append(float(est))
for line in open(DATA_SAVE_PATH+'I(X,L(X)).txt', "r"):
    itr, est = line.split()
    pub_MI_og.append(float(est))

plt.figure()
plt.plot(itr_vec, pub_MI_epoch_first, label='Epoch 1')
plt.plot(itr_vec, pub_MI_epoch_middle, label='Epoch 15')
plt.plot(itr_vec, pub_MI_epoch_last, label='Epoch 50')
plt.plot(itr_vec, pub_MI_og, label='Upper Bound')
plt.ylabel('MI Estimation')
plt.xlabel('Iteration' )
plt.legend()
plt.savefig(FIG_SAVE_PATH+'MI_public.png')

pri_MI_epoch_first = []
pri_MI_epoch_middle = []
pri_MI_epoch_last = []
pri_MI_og = []

for line in open(DATA_SAVE_PATH+'I(T(X),S(X)) epoch 0.txt', "r"):
    itr, est = line.split()
    pri_MI_epoch_first.append(float(est))
for line in open(DATA_SAVE_PATH+'I(T(X),S(X)) epoch 14.txt', "r"):
    itr, est = line.split()
    pri_MI_epoch_middle.append(float(est))
for line in open(DATA_SAVE_PATH+'I(T(X),S(X)) epoch 49.txt', "r"):
    itr, est = line.split()
    pri_MI_epoch_last.append(float(est))
for line in open(DATA_SAVE_PATH+'I(X,S(X)).txt', "r"):
    itr, est = line.split()
    pri_MI_og.append(float(est))

plt.figure()
plt.plot(itr_vec, pri_MI_epoch_first, label='Epoch 1')
plt.plot(itr_vec, pri_MI_epoch_middle, label='Epoch 15')
plt.plot(itr_vec, pri_MI_epoch_last, label='Epoch 50')
plt.plot(itr_vec, pri_MI_og, label='Upper Bound')
plt.ylabel('MI Estimation')
plt.xlabel('Iteration' )
plt.legend()
plt.savefig(FIG_SAVE_PATH+'MI_private.png')

# %%  Analysis and Visualization of InfoShape Data Points
# USe T_SNE method to convert the high-dimensional samples to lower dimensional samples with close distributions

x_pub = train_data_transform_pub_trainedenc.data
y_pub = train_data_transform_pub_trainedenc.targets
x_pri = train_data_transform_pri_trainedenc.data
y_pri = train_data_transform_pri_trainedenc.targets
tsne = TSNE(n_components=2, verbose=1, random_state=seed)
z_pub = tsne.fit_transform(x_pub.cpu())
z_pri = tsne.fit_transform(x_pri.cpu())

# public label
plt.figure()
cdict = {0: 'gold', 1: 'indigo'}

ix1 = np.where(y_pub == 0)
ix2 = np.where(y_pub == 1)

plt.scatter(z_pub[ix1,0], z_pub[ix1,1], c = cdict[0], s=0.1)
plt.scatter(z_pub[ix2,0], z_pub[ix2,1], c = cdict[1], s=0.1)

plt.savefig( FIG_SAVE_PATH + 'infoshape_dataset_pub.png' )

# private label
plt.figure()
cdict = {0: 'deepskyblue', 1: 'darkorange'}

ix1 = np.where(y_pri == 0)
ix2 = np.where(y_pri == 1)

plt.scatter(z_pri[ix1,0], z_pri[ix1,1], c = cdict[0], s=0.1)
plt.scatter(z_pri[ix2,0], z_pri[ix2,1], c = cdict[1], s=0.1)

plt.savefig( FIG_SAVE_PATH + 'infoshape_dataset_pri.png' )

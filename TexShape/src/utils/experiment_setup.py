from .data_structures import ExperimentParams, MINE_Params, EncoderParams
import argparse
import datetime

def parse_args() -> argparse.Namespace:
    # Get args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="utility+privacy",
        choices=["utility", "utility+privacy", "compression", "compression+filtering"],
    )
    parser.add_argument("--num_enc_epochs", type=int, default=10)
    parser.add_argument("--mine_batch_size", type=int, default=-1)
    parser.add_argument("--mine_epochs_privacy", type=int, default=2000)
    parser.add_argument("--mine_epochs_utility", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--use_prev_epochs_mi_model", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="sst2", choices=["sst2", "mnli", "corona"]
    )
    parser.add_argument(
        "--encoder_hidden_sizes",
        type=lambda y: list(map(lambda x: int(x), y.split())),
        default=[512, 256, 128],
    )
    parser.add_argument(
        "--combination_type",
        type=str,
        default="premise_only",
        choices=["concat", "join", "premise_only"],
    )

    return parser.parse_args()


def get_experiment_params(args: argparse.Namespace) -> ExperimentParams:
    # Constants
    experiment_type = args.experiment_type
    num_enc_epochs = args.num_enc_epochs
    mine_batch_size = args.mine_batch_size
    mine_epochs_privacy = args.mine_epochs_privacy
    mine_epochs_utility = args.mine_epochs_utility
    beta = args.beta
    use_prev_epochs_mi_model = args.use_prev_epochs_mi_model
    dataset = args.dataset
    encoder_hidden_sizes = args.encoder_hidden_sizes
    combination_type = args.combination_type

    mine_params = MINE_Params(
        utility_stats_network_model_name="FeedForwardMI",
        utility_stats_network_model_params={"input_size": 128, "output_size": 768},
        privacy_stats_network_model_name="FeedForwardMI3",
        privacy_stats_network_model_params={"input_size": 128},
        mine_batch_size=mine_batch_size,
        mine_epochs_privacy=mine_epochs_privacy,
        mine_epochs_utility=mine_epochs_utility,
        use_prev_epochs_mi_model=use_prev_epochs_mi_model,
        utility_stats_network_model_path=None,
        privacy_stats_network_model_path=None
    )
    
    encoder_params = EncoderParams(
        encoder_hidden_sizes=encoder_hidden_sizes,
        num_enc_epochs=num_enc_epochs,
        enc_save_dir_path=None
    )
    
    experiment_params = ExperimentParams(
        dataset_name=dataset,
        experiment_type=experiment_type,
        mine_args=mine_params,
        encoder_params=encoder_params,
        beta=beta,
        experiment_date=datetime.datetime.now().strftime("%m_%d_%y"),
        combination_type=combination_type,
    )
    
    return experiment_params

if __name__ == "__main__":
        # EXPERIMENT = args.experiment_name
    # TODO: Make this a function
    args = parse_args()
    experiment_params = get_experiment_params(args)
    
    dataset = experiment_params.dataset_name
    experiment_type = experiment_params.experiment_type
    mine_batch_size = experiment_params.mine_args.mine_batch_size
    mine_epochs_privacy = experiment_params.mine_args.mine_epochs_privacy
    use_mi_strategy = experiment_params.mine_args.use_prev_epochs_mi_model
    beta = experiment_params.beta
    encoder_hidden_sizes = experiment_params.encoder_params.encoder_hidden_sizes
    combination_type = experiment_params.combination_type
    now = datetime.datetime.now()
    enc_save_path = f"./encoder_models/{now.strftime('%m_%d_%y')}/{experiment_type}/"
    EXPERIMENT = f"{dataset}-{encoder_hidden_sizes[-1]}-beta={beta}-mine_batch_size={mine_batch_size}-mine_epochs={mine_epochs_privacy}-use_mi_strategy={use_mi_strategy}-combination_type={combination_type}-originalffmi3"
    
    # Encoder params
    # encoded_embedding_size = encoder_hidden_sizes[-1]
    # encoder_model = DenseEncoder3(
    #     768, encoder_hidden_sizes[:-1], encoded_embedding_size
    # ).to(device)

    # Or continue training from a previous model TODO: Fix this
    # encoder_model = torch.load("./encoder_models/MPNET-128-epoch=3.pt")

    # if utility:
    #     stats_network = FeedForwardMI(128, 768).to(device)
    # else:
    #     stats_network = FeedForwardMI3(128).to(device)
    # print("Encoder model: ", encoder_model)

    if use_mi_strategy:
        utility_stats_network_path = f"mi_models/utility/{EXPERIMENT}.pt"
        privacy_stats_network_path = f"mi_models/privacy/{EXPERIMENT}.pt"
    else:
        utility_stats_network_path = None
        privacy_stats_network_path = None
import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn


# FONT_SIZE = 20
FONT_SIZE = 16
sns.set_color_codes()
seaborn.set()

plt.rc('text')
#plt.rcParams['text.latex.preamble'] = [r'\boldmath']

LEGEND_FONT_SIZE = 14
XTICK_LABEL_SIZE = 14
YTICK_LABEL_SIZE = 14

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize': FONT_SIZE,
         'xtick.labelsize': XTICK_LABEL_SIZE,
         'ytick.labelsize': YTICK_LABEL_SIZE,
         'figure.autolayout': True}
pylab.rcParams.update(params)
plt.rcParams["axes.labelweight"] = "bold"

try:
    from .textfile_utils import *
except: 
    from textfile_utils import *

def basic_scatterplot(ts_x = None, ts_y = None, title_str = None, plot_file = None, ylabel = None, lw=3.0, ylim = None, xlabel = 'time', xlim = None, ms=4.0, color = 'b'):

    plt.scatter(ts_x, ts_y, lw=lw, s = ms, color = color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()

# joint plot: scatterplot with the CDFs
def scatter_pdf_plot(ts_x = None, ts_y = None, title_str = None, plot_file = None, ylabel = None, lw=3.0, ylim = None, xlabel = 'time', xlim = None):

    sns.jointplot(x = ts_x, y = ts_y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()


def basic_plot_ts(x,y, ltype, legloc, mtype, title_str = None, plot_file = None, ylabel = None, lw=3.0, ylim = 'none', xlim = 'none', xlabel = 'time', alp=1.0, msize=10, **kwargs):
    # sns.color_palette("rocket", as_cmap=True)
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        ltype (_type_): _description_
        legloc (_type_): _description_
        mtype (_type_): _description_
        title_str (_type_, optional): _description_. Defaults to None.
        plot_file (_type_, optional): _description_. Defaults to None.
        ylabel (_type_, optional): _description_. Defaults to None.
        lw (float, optional): _description_. Defaults to 3.0.
        ylim (str, optional): _description_. Defaults to 'none'.
        xlim (str, optional): _description_. Defaults to 'none'.
        xlabel (str, optional): _description_. Defaults to 'time'.
        alp (float, optional): _description_. Defaults to 1.0.
        msize (int, optional): _description_. Defaults to 10.
        
        **kwargs:
        colors: list of colors for each line
        annotations: list of annotations for each line
        anot_cord: list of annotation coordinates for each line  
        scatter: dict of scatter plot data  
    """
    colors = kwargs.get('colors', None)
    if not colors:
        colors = [item[-1] for item in ltype]
    # Create the plot
    for i in range(len(x)):
        plt.plot(x[i], y[i], ltype[i], color=colors[i],lw=lw,mfc=mtype[i], markersize=msize, alpha=alp)

        # Create annotations if they exist
        anot_cord = kwargs.get('anot_cord', None)
        annotations = kwargs.get('annotations', None)
        if anot_cord:
            for j, txt in enumerate(annotations[i]):
                x_cord = x[i][j]
                y_cord = y[i][j]
                annotation_coordinates = (x_cord+anot_cord[i][j][0], y_cord+anot_cord[i][j][1])
                plt.annotate(txt, (x_cord, y_cord), annotation_coordinates, size=12)
        
        # Add Scatter if it exists
        scatters = kwargs.get('scatter', None)
        if scatters:
            scatter_x = scatters["x"]
            scatter_y = scatters["y"]
            scatter_size = scatters["size"]
            plt.scatter(scatter_x, scatter_y, color="black", s=scatter_size)
            
            if "annotations" in kwargs["scatter"].keys():
                for i, annot in enumerate(kwargs["scatter"]["annotations"]):
                    x_cord = scatter_x[i]
                    y_cord = scatter_y[i]
                    annotation_coordinates = (x_cord+kwargs["scatter"]["annot_cords"][i][0], y_cord+kwargs["scatter"]["annot_cords"][i][1])
                    plt.annotate(annot, (x_cord, y_cord), annotation_coordinates, size=10)
    
    # Add labels
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)

    if ylim != 'none':
        plt.ylim(ylim)
    if xlim != 'none':
        plt.xlim(xlim)

    # Add legend if it exists
    legendd = kwargs.get('legendd', [])
    legend_size = kwargs.get('legend_size', 12)
    if len(legendd) != 0:
        plt.legend(legendd, bbox_to_anchor=legloc, ncol=1, fontsize=legend_size) # len(legendd)

    plt.title(title_str, fontsize=20)
    plt.savefig(plot_file, dpi=400)
    plt.close()


def overlaid_ts(normalized_ts_dict = None, title_str = None, plot_file = None, ylabel = None, xlabel = 'time', fontsize = 30, xticks = None, ylim = None, DEFAULT_ALPHA = 1.0, legend_present = True, DEFAULT_MARKERSIZE = 15, delete_yticks = False):

    # dictionary:
    # key = ts_name, value is a dict, value = {'xvec': , 'ts_vector', 'lw', 'linestyle', 'color'}

    colors = ["denim blue", "medium green", "pale red", "amber", "greyish", "dusty purple"]
    i = 0
    for ts_name, ts_data_dict in normalized_ts_dict.items():
        if 'zorder' in ts_data_dict.keys():
            zorder = ts_data_dict['zorder']
        else:
            zorder = None

        if 'color' in ts_data_dict.keys():
            color = ts_data_dict['color']
        else:
            color = sns.xkcd_rgb[colors[i]]

        if 'alpha' in ts_data_dict.keys():
            alpha = ts_data_dict['alpha']
        else:
            alpha = DEFAULT_ALPHA

        if 'xvec' in ts_data_dict.keys():

            if 'marker' in ts_data_dict.keys():
                plt.plot(ts_data_dict['xvec'], ts_data_dict['ts_vector'], lw= ts_data_dict['lw'], label = ts_name, marker = ts_data_dict['marker'], ls = ts_data_dict['linestyle'], alpha = alpha, ms = DEFAULT_MARKERSIZE, color = color, zorder= zorder)
            else:
                plt.plot(ts_data_dict['xvec'], ts_data_dict['ts_vector'], lw= ts_data_dict['lw'], label = ts_name, ls = ts_data_dict['linestyle'], alpha = alpha, color = color, zorder=zorder)

        else:
            if 'marker' in ts_data_dict.keys():
                plt.plot(ts_data_dict['ts_vector'], lw= ts_data_dict['lw'], label = ts_name, marker = ts_data_dict['marker'], ls = ts_data_dict['linestyle'], alpha = alpha, ms = DEFAULT_MARKERSIZE, color = color, zorder=zorder)
            else:
                plt.plot(ts_data_dict['ts_vector'], lw= ts_data_dict['lw'], label = ts_name, ls = ts_data_dict['linestyle'], alpha = alpha, color = color, zorder=zorder)

        #plt.hold(True)

        i += 1

    if fontsize:
        if xlabel:
            plt.xlabel(xlabel, fontsize=fontsize)
        if ylabel:
            plt.ylabel(ylabel, fontsize=fontsize)
    else:
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

    if xticks:
        plt.xticks(xticks)

    if ylim:
        plt.ylim(ylim)

    if legend_present:
        print('legend present!')
        plt.legend(loc = 'best')


        #if i % 2 == 0:
        #    ncol = 2
        #elif i % 3 == 0:
        #    ncol = 3
        #else:
        #    ncol = 1
        #plt.legend(loc='best', bbox_to_anchor = (0., 1.02, 1., .102), ncol = ncol, fontsize ='x-small')

    if title_str is not None:
        plt.title(title_str, fontsize=fontsize)
    if delete_yticks:
        plt.yticks([])
    plt.savefig(plot_file)
    plt.close()

# plot grid KPI subfigures
def plot_grid(normalized_ts_dict=None,
                    title_str = None,
                    plot_file=None,
                    lw = 3.0,
                    xlabel = None):


    nrow = len(normalized_ts_dict.keys())
    ncol = 1

    plt.close('all')
    f, axarr = plt.subplots(nrow, 1, sharex=True)

    if title_str:
        plt.title(title_str)
    #print(axarr)
    #print(axarr[0])

    row = 0
    for ylabel_name, timeseries_dict in normalized_ts_dict.items():

        if 'x' in timeseries_dict.keys():
            axarr[row].plot(timeseries_dict['x'], timeseries_dict['ts_vector'], lw = lw)
        else:
            axarr[row].plot(timeseries_dict['ts_vector'], lw = lw)

        axarr[row].set_ylabel(ylabel_name)

        if 'ylim' in timeseries_dict.keys():
            axarr[row].set_ylim(timeseries_dict['ylim'])

        if 'xlim' in timeseries_dict.keys():
            axarr[row].set_xlim(timeseries_dict['xlim'])

        if 'yticks' in timeseries_dict.keys():
            if timeseries_dict['yticks']:
                axarr[row].set_yticks(timeseries_dict['yticks'])

        row+= 1

    if xlabel:
        plt.xlabel(xlabel)
    plt.show()
    plt.savefig(plot_file)
    plt.close()


def plot_pdf(data_vector = None, xlabel = None, plot_file = None, title_str = None):

    np_data = np.array(data_vector)

    clean_data = np_data[~np.isnan(np_data)]

    sns.distplot(clean_data)

    plt.xlabel(xlabel)

    plt.title(title_str)

    plt.savefig(plot_file)
    plt.close()




def plot_several_pdf(data_vector_list = None, xlabel = None, plot_file = None, title_str = None, legend = None, ylabel = None, norm = True, xlim=None, kde=False):

    for i, data_vector in enumerate(data_vector_list):
        sns.distplot(data_vector, norm_hist = norm, kde=kde)
        #sns.histplot(data_vector, norm_hist = norm)
        #plt.hold(True)

    plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if title_str:
        plt.title(title_str)
    plt.legend(legend)

    if xlim:
        plt.xlim(xlim)

    plt.savefig(plot_file)
    plt.close()


"""
paired boxplot, hue is a name of a column that controls what to pair by
"""

def plot_paired_boxplot(df = None, x_var = None, y_var = None, plot_file = None, ylim = None, title_str = None, order_list = None, pal = None, hue = None):

    fig = plt.figure()

    if not pal:
        if order_list:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, hue = hue)
        else:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, hue = hue)

    if pal:
        if order_list:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal, hue = hue)
        else:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal, hue = hue)


    if ylim:
        plt.ylim(ylim[0], ylim[1])
    #sns.plt.tight_layout()
    #sns.plt.savefig(plot_file)
    #sns.plt.clf()

    if title_str:
        plt.title(title_str)

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.clf()
    plt.close()



"""
no pairing
"""

def plot_grouped_boxplot(df = None, x_var = None, y_var = None, plot_file = None, ylim = None, title_str = None, order_list = None, pal = None):

    fig = plt.figure()

    if not pal:
        if order_list:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list)
        else:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list)

    if pal:
        if order_list:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal)
        else:
            plot = sns.boxplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal)


    if ylim:
        plt.ylim(ylim[0], ylim[1])
    #sns.plt.tight_layout()
    #sns.plt.savefig(plot_file)
    #sns.plt.clf()

    if title_str:
        plt.title(title_str)

    #plt.tight_layout()
    plt.savefig(plot_file)
    plt.clf()
    plt.close()

def plot_grouped_violinplot(df = None, x_var = None, y_var = None, plot_file = None, ylim = None, title_str = None, order_list = None, pal = None):

    fig = plt.figure()

    if not pal:
        if order_list:
            plot = sns.violinplot(x=x_var, y=y_var, data=df, order = order_list)
        else:
            plot = sns.violinplot(x=x_var, y=y_var, data=df, order = order_list)

    if pal:
        if order_list:
            plot = sns.violinplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal)
        else:
            plot = sns.violinplot(x=x_var, y=y_var, data=df, order = order_list, palette = pal)

    if ylim:
        plt.ylim(ylim[0], ylim[1])

    if title_str:
        plt.title(title_str)

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.clf()



def seaborn_jointplot(x = None, y = None, df = None, title_str = None, plot_file = None):
    sns.jointplot(x=x, y= y , data=df, kind="kde");
    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()


def normalize_ts(input_ts, zero_one_norm = False):

    min_ts = np.min(input_ts)

    max_ts = np.max(input_ts)

    std_ts = np.nanstd(input_ts)

    mean_ts = np.nanmean(input_ts)

    if zero_one_norm:
        normalized_ts = [(x - min_ts)/(max_ts - min_ts) for x in input_ts]
    else:
        normalized_ts = [(x - mean_ts)/std_ts for x in input_ts]

    return normalized_ts

if __name__ == "__main__":
    print('hello')
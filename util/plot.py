import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from util import get_logger
logger = get_logger(__name__)

def plot_mat(df: pd.DataFrame, keys, png_dpath):
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)

    for key in keys:
        plt.figure()
        # logger.info(df)
        id_list = df["dialog_id"]# df["episode_id"]
        # logger.info(episode_list)
        values = df[key]
        values_rolling = df[key].rolling(window=100, min_periods=10).mean()
        y_min = max([values.min(), values_rolling.min()-0.1])
        y_max = min([values.max(), values_rolling.max()+0.1])
        try:
            plt.ylim(y_min, y_max)
        except ValueError:
            pass
        plt.plot(id_list, values)
        plt.plot(id_list, values_rolling)
        plt.savefig(os.path.join(png_dpath, key+".png"))
        plt.close("all")
    logger.info("Saved slot results to {}".format(png_dpath))

def plot_sns(df: pd.DataFrame, keys, png_dpath):
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)
        
    for key in keys:
        sns.set(rc={'figure.facecolor':'white'})
        sns.lineplot(data=df.loc[:, ['iteration_id', key]], x='iteration_id', y=key)
        logger.info("Plotting {}".format(key))
        if key == 'task_success':
            plt.ylim(0.2, 0.9)
        plt.savefig(os.path.join(png_dpath, key+".png"))
        plt.close("all")
    logger.info("Saved slot results to {}".format(png_dpath))

def plot_sns_f1s(df, f1_keys, png_dpath):
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(5,3))
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.2, top=0.9)

    y_min, y_max = 0.55, 0.75
    title = "F1 Score"
    for key in f1_keys:
        sns.lineplot(data=df, x='iteration_id', y=key, ax=ax)
    # ax.legend(labels=["BERT NLU", "PPN for BERT NLU"], loc="lower right", fontsize=18)
    ax.legend(labels=["BERT NLU"], loc="lower right", fontsize=18)
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel(title, fontsize=18)
    # ax.set_title(title, fontsize=20)
    ax.set_ylim(y_min, y_max)
    plt.savefig(os.path.join(png_dpath, "+".join(f1_keys)+".pdf"))
    plt.savefig(os.path.join(png_dpath, "+".join(f1_keys)+".png"))
    plt.close("all")

def sub_plot_sns(df, keys, png_dpath):
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)

    sns.set(style="darkgrid")
    fig, axs3 = plt.subplots(1, 3, figsize=(18,5))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9)
    assert len(axs3) == len(keys)
    for i, key in enumerate(keys):
        logger.info("Plotting {}".format(key))
        if key == "task_success":
            y_min, y_max = 0.2, 0.8
            title = "Task Success"
            y_label = "Success Rate"
        elif key == "inform_F1":
            y_min, y_max = 0.2, 0.8
            title = "Inform F1"
            y_label = "F1 Score"
        elif key == "book_rate":
            y_min, y_max = 0.2, 0.8
            title = "Match Rate"
            y_label = "Match Rate"
        elif key == "turn":
            y_min, y_max = 4, 14
            title = "Turn"
            y_label = "Average Turn"
        else:
            title = key

        sns.lineplot(data=df, x='iteration_id', y=key, hue='strategy', ax=axs3[i])
        axs3[i].legend(loc="lower right", fontsize=18)
        axs3[i].set_xlabel("Iteration", fontsize=18)
        axs3[i].set_ylabel(y_label, fontsize=18)
        axs3[i].set_ylim(y_min, y_max)
        axs3[i].set_title(title, fontsize=20)
        axs3[i].tick_params(labelsize = 18)
    plt.savefig(os.path.join(png_dpath,"+".join(keys) +".png"))
    plt.savefig(os.path.join(png_dpath,"+".join(keys) +".pdf"))
    plt.close("all")

def multi_plot_sns(df, keys, png_dpath):
    if not os.path.exists(png_dpath):
        os.makedirs(png_dpath)
    
    for key in keys:
        logger.info("Plotting {}".format(key))
        # if key == "task_success":
        #     y_min, y_max = 0.3, 0.9
        #     title = "Task Success"
        # elif key == "inform_F1":
        #     y_min, y_max = 0.3, 0.9
        #     title = "Inform F1"
        # elif key == "book_rate":
        #     y_min, y_max = 0.3, 0.9
        #     title = "Match Rate"
        # elif key == "turn":
        #     y_min, y_max = 5, 15
        #     title = "Turn"
        # else:
        title = key

        sns.set(rc={'figure.facecolor':'white'})
        ax = sns.lineplot(data=df, x='iteration_id', y=key, hue='strategy')
        ax.legend(loc="lower right", fontsize=18)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("")
        # ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=20)
        plt.savefig(os.path.join(png_dpath, key+".png"))
        plt.close("all")
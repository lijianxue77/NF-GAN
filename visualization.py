import matplotlib.pyplot as plt
import seaborn as sns
from utils import *


def plot_losses(path_d, path_g):
    df_d, df_g = build_losses_dfs(path_d, path_g)
    plot_loss(df_d, 'd_losses')
    plot_loss(df_g, 'g_losses', color='darkorange')


def plot_loss(losses, fig_name='losses', color='forestgreen'):
    """generate the figure of losses"""
    plt.style.use('seaborn-colorblind')
    plt.plot(losses, color=color, label=fig_name)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend(loc='upper right')
    plt.savefig('./stats/{}.png'.format(fig_name))
    plt.clf()


def plot_shared_shaded(path_list, use_target_list, r_columns, f_columns):
    df_list = build_traffic_dfs(path_list, use_target_list, r_columns, f_columns)
    df_r = df_list[0]
    df_f = df_list[1]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes = axes.flatten()
    sns.kdeplot(ip_transform(df_r['srcip']), shade=True, ax=axes[0], color='blue')
    sns.kdeplot(ip_transform(df_f['srcip']), shade=True, ax=axes[0], color='red', linestyles="--")
    axes[0].set(ylabel='UNSW-NB15', xlabel='Source IP')
    sns.kdeplot(df_r['sport'], shade=True, ax=axes[1], color='blue')
    sns.kdeplot(df_f['sport'], shade=True, ax=axes[1], color='red', linestyles="--")
    axes[1].set(ylabel='', xlabel='Source Port')
    sns.kdeplot(ip_transform(df_r['dstip']), shade=True, ax=axes[2], color='blue')
    sns.kdeplot(ip_transform(df_f['dstip']), shade=True, ax=axes[2], color='red', linestyles="--")
    axes[2].set(ylabel='', xlabel='Destination IP')
    sns.kdeplot(df_r['dsport'], shade=True, ax=axes[3], color='blue', label='Real Network Flow')
    sns.kdeplot(df_f['dsport'], shade=True, ax=axes[3], color='red', linestyles="--", label='Generated Network Flow')
    axes[3].set(ylabel='', xlabel='Destination Port')

    plt.tight_layout()
    plt.savefig('./stats/kde_density.png')
    plt.clf()
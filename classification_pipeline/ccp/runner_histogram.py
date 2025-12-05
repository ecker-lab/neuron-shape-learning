import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from cmcrameri import cm
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse


color_dict_layer = {
    'L23': cm.batlow.colors[0],
    'L4': cm.batlow.colors[50],
    'L5': cm.batlow.colors[100],
    'L6': cm.batlow.colors[150],
}


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--datapath', type=str, default='/scratch/usr/nimpede1/pcs_v7_volume_rotation', help='path where data is stored')
    return parser.parse_args()

def main(args):
    df = pd.read_pickle(f'{args.log_dir}/df_embeddings_tsne_classifier.pkl')
    label_df = pd.read_pickle(f'{args.datapath}/graphdino_assigned_layer.pkl')
    df = df.merge(label_df, on='segment_split')
    
    # plot embeddings
    out_path = Path(args.log_dir, 'tsne')
    out_path.mkdir(parents=True, exist_ok=True)

    plot_layer_histograms(df, palette=color_dict_layer, out_path=out_path)



def plot_layer_histograms(df,  palette="Set2", out_path='hist.pdf'):
    fig, axes = plt.subplots(1, 2, figsize=(2, 1), sharey=True)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'

    label_type = "layer_prediction"
    labels = sorted(df[label_type].dropna().unique())

    sns.countplot(
        x=label_type, 
        data=df,
        hue=label_type,
        hue_order=labels,
        palette=palette, 
        order=labels,
        width=0.95,
        ax=axes[0]
    )

    # Remove x-axis only
    axes[0].set_xlabel("")
    axes[0].set_xticks([])
    axes[0].set_yticks([0, 5000, 10000])
    
    # Keep y-axis visible
    axes[0].set_ylabel("Neurons")
    # Format y-axis with commas
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    if axes[0].get_legend():
        axes[0].get_legend().remove()


    # Remove the box (spines) around the plot except left spine
    for spine in ["top", "right", "bottom"]:
        axes[0].spines[spine].set_visible(False)

    label_type = "assigned_layer"
    labels = sorted(df[label_type].dropna().unique())

    sns.countplot(
        x=label_type, 
        data=df,
        hue=label_type,
        hue_order=labels,
        palette=palette, 
        order=labels,
        width=0.95,
        ax=axes[1]
    )

    # Remove x-axis only
    axes[1].set_xlabel("")
    axes[1].set_xticks([])
    
    # Keep y-axis visible
    axes[1].set_ylabel("Neurons")

    if axes[1].get_legend():
        axes[1].get_legend().remove()


    # Remove the box (spines) around the plot except left spine
    for spine in ["top", "right", "bottom"]:
        axes[1].spines[spine].set_visible(False)


    plt.tight_layout()
    fig.savefig(f'{out_path}/hist.pdf', bbox_inches='tight', transparent=True, dpi=300)


if __name__ == '__main__':
    args = parse_args()
    main(args)

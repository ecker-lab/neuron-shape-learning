import os
import numpy as np
from openTSNE import TSNE
import matplotlib
import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap
import pacmap

color_dict = dict(
    {
        '23P': 'goldenrod',
        '4P': 'steelblue',
        '5P-IT': 'indianred',
        '5P-PT': 'purple',
        '5P-NP': 'darkred',
        '6P-CT': 'teal',
        '6P-IT': 'darkgreen',
        'BC': 'midnightblue',
        'BPC': 'royalblue',
        'MC': 'cadetblue',
        'NGC': 'slateblue',
        'None': 'grey',
    }
)


dimreds = ['tsne', 'umap', 'pacmap']
ps = [30, 100, 300, 1000]


def cluster_net(data_dir, log_dir):
    # plot embeddings
    out_path = Path(log_dir, 'ablations')
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists('%s/df_embeddings_tsne_classifier_ablations_parameter.pkl' % out_path):
        if not os.path.exists('%s/df_embeddings_tsne_classifier.pkl' % log_dir):
            print('Run cluster first!')
            exit()
        else:
            df = pd.read_pickle('%s/df_embeddings_tsne_classifier.pkl' % log_dir)

            for p in ps:
                print('Extract umap embeddings')
                df = extract_umap_embeddings(df, p, seed=42)
                print('Extract PaCMAP embeddings')
                df = extract_pacmap_embeddings(df, p, seed=42)
                print('Extract tsne embeddings')
                df = extract_tsne_embeddings(df, p, seed=42)
            print('Finished extracting embeddings')
            
            df.to_pickle(os.path.join(out_path, 'df_embeddings_tsne_classifier_ablations_parameter.pkl'))
    else:
        df = pd.read_pickle('%s/df_embeddings_tsne_classifier_ablations_parameter.pkl' % out_path)
    
    # plot EXC plot_expert_labels with tsne, PACMAP and UMAP
    df = ablation_dimred(df, data_dir, out_path, filename='ablation_soma_y_v2')
   

def ablation_dimred(df, data_dir, out_path, filename):
    print('Cluster start ... ')
    # plot only exc cells
    df = df[(df['ei_prediction'] == 'exc')]

    label_df = pd.read_pickle('%s/graphdino_assigned_layer.pkl' % data_dir)
    df = df.merge(label_df, on='segment_split')

    # Convert cm to inches for matplotlib (1 inch = 2.54 cm)
    fig_width = 15 / 2.54  # ~5.9 inches
    fig_height = 12 / 2.54  # ~7.9 inches
    
    fig, axes = plt.subplots(len(dimreds), len(ps), figsize=(fig_width, fig_height))
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'
    # axes = axes.flatten()  # make indexing easier
    
    for i, dimred in enumerate(dimreds):
        for j, p in enumerate(ps):
            ax = axes[i, j]
            # plot_expert_labels(df=df, label_type='gt_labels', ax=ax, dimred=dimred, p=p)
            plot_pred(df=df, label_type='soma_y', ax=ax, dimred=dimred, p=p)
            # remove legend except for the first subplot
            ax.get_legend().remove()
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # leave space for legend
    fig.savefig(f'{out_path}/{filename}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close(fig)

    return df


def extract_tsne_embeddings(df, perplexity=300, seed=42):
    latents = np.stack(df['embeddings'].values).astype(float)

    tsne = TSNE(
        perplexity=perplexity,
        metric='euclidean',
        n_jobs=8,
        random_state=seed,
        verbose=False,
    )

    tsne_emb = tsne.fit(latents)
    tsnes = np.array(list(tsne_emb))
    df[f'tsne_x_{perplexity}'] = tsnes[:,0]
    df[f'tsne_y_{perplexity}'] = tsnes[:,1]

    return df


def extract_umap_embeddings(df, n_neighbors=15, seed=42):
    latents = np.stack(df['embeddings'].values).astype(float)

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,     # adjust for local/global structure
        min_dist=0.1,                # how tightly points cluster
        n_components=2,
        metric="euclidean",
        random_state=seed
    )

    umap_embeds = umap_model.fit_transform(latents)
    df[f'umap_x_{n_neighbors}'] = umap_embeds[:,0]
    df[f'umap_y_{n_neighbors}'] = umap_embeds[:,1]

    return df


def extract_pacmap_embeddings(df, n_neighbors=10, seed=42):
    latents = np.stack(df['embeddings'].values).astype(float)

    pacmap_model = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors, 
        MN_ratio=0.5, 
        FP_ratio=2.0
    )

    pacmap_embeds = pacmap_model.fit_transform(latents)
    df[f'pacmap_x_{n_neighbors}'] = pacmap_embeds[:, 0]
    df[f'pacmap_y_{n_neighbors}'] = pacmap_embeds[:, 1]

    return df


def plot_expert_labels(df, label_type, ax, dimred='tsne', p='perplexity'):
    '''
    plot colored expert labels and unlabeled data in background
    '''
    labels = sorted(df[label_type].dropna().unique())

    size = 1

    sns.scatterplot(
        data=df[df['mode'] != 'test'],
        x=f'{dimred}_x_{p}',
        y=f'{dimred}_y_{p}',
        hue=label_type,
        ax=ax,
        palette=color_dict,
        alpha=0.2,
        linewidth=0,
        legend='full',
        rasterized=True,
        s=size,
    )

    sns.scatterplot(
        data=df[df['mode'] == 'test'],
        x=f'{dimred}_x_{p}',
        y=f'{dimred}_y_{p}',
        hue=label_type,
        hue_order=labels[:-1],
        ax=ax,
        palette=color_dict,
        alpha=1.0,
        linewidth=0,
        legend='full',
        rasterized=True,
        s=size,
    )

    ax.axis('off')
    ax.set_aspect('equal')


def plot_pred(df, label_type, ax, dimred='tsne', p='perplexity', palette=cm.batlow):

    labels = sorted(df[label_type].dropna().unique())

    sns.scatterplot(
        data=df,
        x=f'{dimred}_x_{p}',
        y=f'{dimred}_y_{p}',
        hue=label_type,
        hue_order=labels,
        ax=ax,
        palette=palette,
        alpha=0.75,
        linewidth=0,
        rasterized=True,
        s=1,
    )

    ax.axis('off')
    ax.set_aspect('equal')
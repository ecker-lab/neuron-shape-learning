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


def cluster_net(log_dir):
    print('Cluster start ... ')

    # plot embeddings
    out_path = Path(log_dir, 'ablations')
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists('%s/df_embeddings_tsne_classifier_ablations.pkl' % out_path):
        if not os.path.exists('%s/df_embeddings_tsne_classifier.pkl' % log_dir):
            print('Run cluster first!')
            exit()
        else:
            df = pd.read_pickle('%s/df_embeddings_tsne_classifier.pkl' % log_dir)
    else:
        df = pd.read_pickle('%s/df_embeddings_tsne_classifier_ablations.pkl' % out_path)
    
    # plot EXC plot_expert_labels with tsne with 10 seeds
    df = ablation_tsne(df, out_path, filename='01_ablation_tsne')
    # plot EXC plot_expert_labels with PCA and UMAP
    df = ablation_dimred(df, out_path, filename='02_ablation_dimred')
   

def ablation_dimred(df, out_path, filename):
    print('Extract umap embeddings')
    df = extract_umap_embeddings(df, seed=42)
    print('Extract pca embeddings')
    df = extract_pca_embeddings(df, seed=42)
    
    df.to_pickle(os.path.join(out_path, 'df_embeddings_tsne_classifier_ablations.pkl'))
    
    # plot only exc cells
    df = df[(df['ei_prediction'] == 'exc')]
    # print(f'Number of cells predicted as excitatory: {len(df)}')
    df = df[(df['ei'] != 'inh')]
    # print(f'Number of cells after filtering by Weis et al. (2025): {len(df)}')

    # Convert cm to inches for matplotlib (1 inch = 2.54 cm)
    fig_width = 15 / 2.54  # ~5.9 inches
    fig_height = 4 / 2.54  # ~7.9 inches
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'
    axes = axes.flatten()  # make indexing easier
    
    dimreds = ['umap', 'pca']

    for i, ax in enumerate(tqdm(axes)):
        if i < len(dimreds):
            plot_expert_labels(df=df, label_type='gt_labels', ax=ax, dimred=dimreds[i], seed=42)
            # remove legend except for the first subplot
            ax.get_legend().remove()
        else:
            ax.axis("off")  # empty subplot if not enough data
    
    # Move legend from first plot to the left side
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0, 0.5))
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # leave space for legend
    fig.savefig(f'{out_path}/{filename}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close(fig)

    return df


def ablation_tsne(df, out_path, filename):
    seeds = [1337, 0, 9, 16, 50, 81, 110, 66, 6, 13]
    
    print('Extract tsne embeddings for 10 seeds')
    for seed in tqdm(seeds):
        df = extract_tsne_embeddings(df, seed)
    
    df.to_pickle(os.path.join(out_path, 'df_embeddings_tsne_classifier_ablations.pkl'))
    
    # plot only exc cells
    df = df[(df['ei_prediction'] == 'exc')]
    # print(f'Number of cells predicted as excitatory: {len(df)}')
    df = df[(df['ei'] != 'inh')]
    # print(f'Number of cells after filtering by Weis et al. (2025): {len(df)}')

    # Convert cm to inches for matplotlib (1 inch = 2.54 cm)
    fig_width = 15 / 2.54  # ~5.9 inches
    fig_height = 20 / 2.54  # ~7.9 inches
    
    fig, axes = plt.subplots(5, 2, figsize=(fig_width, fig_height))
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'
    axes = axes.flatten()
    
    for i, ax in enumerate(tqdm(axes)):
        if i < len(seeds):
            plot_expert_labels(df=df, label_type='gt_labels', ax=ax, seed=seeds[i])
            # remove legend except for the first subplot
            ax.get_legend().remove()
        else:
            ax.axis("off")  # empty subplot if not enough data
    
    # Move legend from first plot to the left side
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0, 0.5))
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # leave space for legend
    fig.savefig(f'{out_path}/{filename}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close(fig)

    return df


def extract_tsne_embeddings(df, seed=42):
    perplexity = 300
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
    df[f'tsne_x_{seed}'] = tsnes[:,0]
    df[f'tsne_y_{seed}'] = tsnes[:,1]

    return df


def extract_umap_embeddings(df, seed=42):
    latents = np.stack(df['embeddings'].values).astype(float)

    umap_model = umap.UMAP(
        n_neighbors=15,     # adjust for local/global structure
        min_dist=0.1,       # how tightly points cluster
        n_components=2,
        metric="euclidean",
        random_state=seed
    )

    umap_embeds = umap_model.fit_transform(latents)
    df[f'umap_x_{seed}'] = umap_embeds[:,0]
    df[f'umap_y_{seed}'] = umap_embeds[:,1]

    return df


def extract_pca_embeddings(df, seed=42):
    latents = np.stack(df['embeddings'].values).astype(float)

    pca_model = PCA(
        n_components=2,
        random_state=seed
    )

    pca_embeds = pca_model.fit_transform(latents)
    df[f'pca_x_{seed}'] = pca_embeds[:, 0]
    df[f'pca_y_{seed}'] = pca_embeds[:, 1]

    return df


def plot_expert_labels(df, label_type, ax, dimred='tsne', seed=42):
    '''
    plot colored expert labels and unlabeled data in background
    '''
    labels = sorted(df[label_type].dropna().unique())

    sns.scatterplot(
        data=df[df['mode'] != 'test'],
        x=f'{dimred}_x_{seed}',
        y=f'{dimred}_y_{seed}',
        hue=label_type,
        ax=ax,
        palette=color_dict,
        alpha=0.2,
        linewidth=0,
        legend='full',
        rasterized=True,
        s=20,
    )

    sns.scatterplot(
        data=df[df['mode'] == 'test'],
        x=f'{dimred}_x_{seed}',
        y=f'{dimred}_y_{seed}',
        hue=label_type,
        hue_order=labels[:-1],
        ax=ax,
        palette=color_dict,
        alpha=1.0,
        linewidth=0,
        legend='full',
        rasterized=True,
        s=20,
    )

    ax.axis('off')
    ax.set_aspect('equal')

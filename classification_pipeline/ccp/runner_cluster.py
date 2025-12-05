import os
import numpy as np
from openTSNE import TSNE
import matplotlib
import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns
import pandas as pd
from pathlib import Path

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

color_dict_exc = dict(
    {
        '23P': 'goldenrod',
        '4P': 'steelblue',
        '5P-IT': 'indianred',
        '5P-PT': 'purple',
        '5P-NP': 'darkred',
        '6P-CT': 'teal',
        '6P-IT': 'darkgreen',
        'None': 'grey',
    }
)

color_dict_layer = {
    'L23': cm.batlow.colors[0],
    'L4': cm.batlow.colors[50],
    'L5': cm.batlow.colors[100],
    'L6': cm.batlow.colors[150],
}

color_dict_ei = {
    'inh': "#D4A838",  # orange-yellow
    'exc': '#D55E00',  # orange-red
    'None': "#949494", # grey
}

int2label = {
    0:'23P',
    1:'4P',
    2:'5P-IT',
    3:'5P-NP',
    4:'5P-PT',
    5:'6P-CT',
    6:'6P-IT',
    7:'BC',
    8:'BPC',
    9:'MC',
    10:'NGC',
    11:'None'
}

def cluster_net(log_dir, data_path):
    print('Cluster start ... ')

    classifier_path = Path(log_dir, 'classifier')

    if not os.path.exists('%s/df_classifier.pkl' % classifier_path):
        print('Run classification first!')
        exit()
    else:
        if not os.path.exists('%s/df_embeddings_tsne_classifier.pkl' % log_dir):
            df = pd.read_pickle('%s/df_classifier.pkl' % classifier_path)
            df = extract_tsne_embeddings(df)
            # tsne for column data
            df = extract_tsne_embeddings_column(df)
            df = gt_pred_to_label(df)
            df = extract_ei_label(df)
            df.to_pickle(os.path.join(log_dir, 'df_embeddings_tsne_classifier.pkl'))
        else:
            df = pd.read_pickle('%s/df_embeddings_tsne_classifier.pkl' % log_dir)
    
    # plot embeddings
    out_path = Path(log_dir, 'tsne')
    out_path.mkdir(parents=True, exist_ok=True)
    
    # all cells GT
    plot_expert_labels(df=df, label_type='gt_labels', title='Expert cell types', experiment_dir=out_path)
    # column data GT
    plot_column_data_only(df=df, label_type='gt_labels', title='Column data', experiment_dir=out_path, tsne='tsne')
    plot_column_data_only(df=df, label_type='gt_labels', title='Column data', experiment_dir=out_path, tsne='tsne_column')
    plot_column_data_only(df=df, label_type='ei', title='Column data Exc/Inh', experiment_dir=out_path, tsne='tsne_column', palette=color_dict_ei)
    # all cells model prediction cell types
    plot_pred(df=df, label_type='pred_labels', title='Model prediction', experiment_dir=out_path, palette=color_dict)
    # all cells classifier prediction cell types
    plot_pred(df=df, label_type='cell_type_prediction', title='Classifier Prediction', experiment_dir=out_path, palette=color_dict)
    # all cells classifier prediction inh/exc
    plot_pred(df=df, label_type='ei_prediction', title='Classifier Prediction', experiment_dir=out_path, palette=color_dict_ei)

    # EXCITATORY ANALYSIS (following Weis et al. 2024)
    df = df[(df['ei_prediction'] == 'exc')]
    df = df[(df['ei'] != 'inh')]

    # exc cells GT
    plot_expert_labels(df=df, label_type='gt_labels', title='Expert cell types', experiment_dir=out_path, exc_only=True)
    plot_column_data_only(df=df, label_type='gt_labels', title='Column data', experiment_dir=out_path, tsne='tsne_column', exc_only=True)
    # exc cells classifier prediction layer
    plot_pred(df=df, label_type='layer_prediction', title='Classifier Prediction', experiment_dir=out_path, palette=color_dict_layer)
    # exc cells classifier prediction cell types
    plot_pred(df=df, label_type='exc_cell_type_prediction', title='Classifier Prediction', experiment_dir=out_path, palette=color_dict_exc)

    # EXCITATORY ANALYSIS (labels produced by Weis et al. 2024)
    label_df = pd.read_pickle('%s/graphdino_assigned_layer.pkl' % data_path)
    df = df.merge(label_df, on='segment_split')
    # exc cells assigned layer
    plot_pred(df=df, label_type='assigned_layer', title='Cortical layer', experiment_dir=out_path, palette=color_dict_layer)
    # exc cells soma depth
    plot_pred(df=df, label_type='soma_y', title='Soma Depth', experiment_dir=out_path, palette=cm.batlow)


def extract_tsne_embeddings(df):
    perplexity = 300

    latents = np.stack(df['embeddings'].values).astype(float)

    print(f'Extract tsne embeddings for {len(latents)} cells with perplexity {perplexity}...')

    tsne = TSNE(
        perplexity=perplexity,
        metric='euclidean',
        n_jobs=8,
        random_state=42,
        verbose=False,
    )

    tsne_emb = tsne.fit(latents)
    tsnes = np.array(list(tsne_emb))
    tsne_x, tsne_y = tsnes[:,0], tsnes[:,1]
    df[f'tsne_x'] = tsne_x
    df[f'tsne_y'] = tsne_y

    return df


def extract_tsne_embeddings_column(df):
    perplexity = 30
        
    df['tsne_column_x'] = np.nan
    df['tsne_column_y'] = np.nan
    subset = df[df['mode'] == 'test']
    latents = np.stack(subset['embeddings'].values).astype(float)

    print(f'Extract tsne embeddings for {len(latents)} cells with perplexity {perplexity}...')

    tsne = TSNE(
        perplexity=perplexity,
        metric='euclidean',
        n_jobs=8,
        random_state=42,
        verbose=False,
    )

    tsne_emb = tsne.fit(latents)
    tsnes = np.array(list(tsne_emb))
    tsne_x, tsne_y = tsnes[:,0], tsnes[:,1]
    df.loc[df['mode'] == 'test', 'tsne_column_x'] = tsne_x
    df.loc[df['mode'] == 'test', 'tsne_column_y'] = tsne_y

    return df
    

def gt_pred_to_label(df):
    gts = np.stack(df['gt'].values)
    preds = np.stack(df['pred'].values)
    gt_labels = [int2label[c] for c in gts]
    pred_labels = [int2label[c] for c in preds]
    df['gt_labels'] = gt_labels
    df['pred_labels'] = pred_labels

    return df


def extract_ei_label(df):
    df['ei'] = 'none'
    df.loc[df['gt'] < 11, 'ei'] = 'inh'
    df.loc[df['gt'] < 7, 'ei'] = 'exc'

    return df


def plot_pred(df, label_type, title, experiment_dir, palette=cm.batlow):
    print(f'Plot {title}...')
    labels = sorted(df[label_type].dropna().unique())

    fig, ax = plt.subplots(1,1)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'

    sns.scatterplot(
        data=df,
        x='tsne_x',
        y='tsne_y',
        hue=label_type,
        hue_order=labels,
        ax=ax,
        palette=palette,
        alpha=0.75,
        linewidth=0,
        rasterized=True,
        s=20,
    )

    ax.axis('off')
    ax.set_aspect('equal')

    plt.legend(title='', bbox_to_anchor=(1, 1));
    fig.savefig(f'{experiment_dir}/{label_type}.pdf', bbox_inches='tight', transparent=True, dpi=300)


def plot_expert_labels(df, label_type, title, experiment_dir, exc_only=False):
    '''
    plot colored expert labels and unlabeled data in background
    '''
    print(f'Plot {title}...')
    labels = sorted(df[label_type].dropna().unique())

    fig, ax = plt.subplots(1,1)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'

    sns.scatterplot(
        data=df[df['mode'] != 'test'],
        x='tsne_x',
        y='tsne_y',
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
        x='tsne_x',
        y='tsne_y',
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

    label_type = f'{label_type}_exc' if exc_only else label_type

    plt.legend(title='', bbox_to_anchor=(1, 1));
    fig.savefig(f'{experiment_dir}/{label_type}.pdf', bbox_inches='tight', transparent=True, dpi=300)


def plot_column_data_only(df, label_type, title, experiment_dir, tsne='tsne', palette=color_dict, exc_only=False):
    '''
    plot only labeled test set
    '''
    print(f'Plot {title}...')

    labels = sorted(df[label_type].dropna().unique())

    fig, ax = plt.subplots(1,1)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['font.size'] = 8
    matplotlib.rcParams['svg.fonttype'] = 'none'

    sns.scatterplot(
        data=df[df['mode'] == 'test'],
        x=f'{tsne}_x',
        y=f'{tsne}_y',
        hue=label_type,
        hue_order=labels[:-1],
        ax=ax,
        palette=palette,
        # style='mode_finetune',
        alpha=1.0,
        linewidth=0,
        legend='full',
        rasterized=True,
        s=20,
    )

    ax.axis('off')
    ax.set_aspect('equal')

    label_type = f'{label_type}_exc' if exc_only else label_type

    plt.legend(title='', bbox_to_anchor=(1, 1));
    fig.savefig(f'{experiment_dir}/{label_type}_labeled_testset_{tsne}.pdf', bbox_inches='tight', transparent=True, dpi=300)

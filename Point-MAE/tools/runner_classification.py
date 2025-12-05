import torch
from tools import builder
from utils import misc
import os
from utils.logger import *
from datasets import build_dataset_from_cfg

import numpy as np
import pandas as pd
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True
logging.getLogger().setLevel(logging.INFO)

from ccp import classify_run_net


def extract_embeddings(config, base_model, dataloader, num_class=12, vote_num=3):
    embeddings, targets, predictions, segment_splits = [], [], [], []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            points = data[0].cuda()
            points = misc.fps(points, npoints)
            target = data[1]

            for t in target:
                targets.append(t)

            vote_pool = torch.zeros(target.size()[0], num_class).cuda()

            for _ in range(vote_num):
                pred, embs = base_model(points, vis=True)
                vote_pool += pred
            pred = vote_pool / vote_num
            pred_choice = pred.data.max(1)[1]
            
            embs = embs.cpu().numpy()
            preds = pred_choice.cpu().numpy()
            for pred, emb, model_id in zip(preds, embs, model_ids):
                embeddings.append(emb)
                predictions.append(pred)
                segment_splits.append(model_id)
    return np.array(segment_splits), np.array(embeddings), np.array(targets), np.array(predictions)


def run_embedding_extraction(args, config, logger):
    base_model = builder.model_builder(config.model)

    # TODO FOR EMBEDDINGS FROM FINETUNED MODEL COMMENT IN THE FOLLOWING BLOCK #############
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    #######################################################################################

    # TODO FOR EMBEDDINGS FROM PRETRAINED MODEL COMMENT IN THE FOLLOWING BLOCK ############
    # resume ckpts 
    # if args.resume:
    #     start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
    #     best_metrics = Acc_Metric(best_metrics)
    # else:
    #     if args.ckpts is not None:
    #         base_model.load_model_from_ckpt(args.ckpts)
    #     else:
    #         print_log('Training from scratch', logger = logger)
    #######################################################################################

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    base_model.eval()  # set model to eval mode

    columns=['segment_split', 'embeddings', 'mode', 'gt', 'pred']
    df = pd.DataFrame(columns=columns)

    config.dataset_finetune.train.others.bs = config.total_bs
    config.dataset_finetune.val.others.bs = 1
    config.dataset_finetune.test.others.bs = 1

    '''DATA LOADING'''
    for mode in ['val', 'train']:
        print_log(f'Load dataset {mode}...')
        
        dataset_config = getattr(config.dataset, mode)
        dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config.others.bs,
                                                shuffle = False, 
                                                drop_last = False,
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=misc.worker_init_fn)

        with torch.no_grad():
            segment_splits, embeddings, gt_labels, preds = extract_embeddings(config, base_model, dataloader, num_class=12)
            print_log(f'Embeddings for {mode} set are extracted. {segment_splits.shape=}, {embeddings.shape=}, {gt_labels.shape=}, {preds.shape=}')

        df_mode = pd.DataFrame(data={'segment_split':list(segment_splits),'embeddings':list(embeddings),'mode': [mode]*len(embeddings), 'mode_finetune':['None']*len(embeddings), 'gt':gt_labels, 'pred':preds})
        df = pd.concat([df, df_mode], axis=0)

    for mode_finetune in ['val', 'train', 'test']:
        print_log(f'Load dataset {mode_finetune} of test dataset...')
        
        dataset_config = getattr(config.dataset_finetune, mode_finetune)
        dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config.others.bs,
                                                shuffle = False, 
                                                drop_last = False,
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=misc.worker_init_fn)

        with torch.no_grad():
            segment_splits, embeddings, gt_labels, preds = extract_embeddings(config, base_model, dataloader, num_class=12)
            print_log(f'Embeddings for {mode} set are extracted. {segment_splits.shape=}, {embeddings.shape=}, {gt_labels.shape=}, {preds.shape=}')

        df_mode = pd.DataFrame(data={'segment_split':list(segment_splits),'embeddings':list(embeddings),'mode': ['test']*len(embeddings), 'mode_finetune':[mode_finetune]*len(embeddings), 'gt':gt_labels, 'pred':preds})
        df = pd.concat([df, df_mode], axis=0)

    return df


def classify_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Classification start ... ', logger = logger)

    data_path = Path('./vis_neurons', args.exp_name)
    data_path.mkdir(parents=True, exist_ok=True)

    out_path = Path(data_path, 'classifier')
    out_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists('%s/embeddings.pkl' % data_path):
        df = run_embedding_extraction(args, config, logger)
        df.to_pickle('%s/embeddings.pkl' % data_path)

    classify_run_net(data_path, config.dataset.train._base_.PC_PATH)
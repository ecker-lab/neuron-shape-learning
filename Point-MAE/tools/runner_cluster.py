import os

from ccp import cluster_run_net

def cluster_net(args, config):
    print('Cluster start ... ')

    log_dir = os.path.join('./vis_neurons', args.exp_name)
    data_path = config.dataset.train._base_.PC_PATH

    cluster_run_net(log_dir, data_path)
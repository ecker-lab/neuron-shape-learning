import argparse

from ccp import cluster_run_net_ablation, cluster_run_net_ablation_parameter


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    return parser.parse_args()


def main(args):
    experiment_dir = 'vis_neurons/' + args.log_dir
    data_path = '/scratch/usr/nimpede1/pcs_v7_volume_rotation'
    # cluster_run_net_ablation(experiment_dir)
    # cluster_run_net_ablation(data_path, experiment_dir)
    cluster_run_net_ablation_parameter(data_path, experiment_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

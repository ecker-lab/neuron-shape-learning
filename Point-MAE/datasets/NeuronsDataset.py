import os
from os.path import join
from torch.utils.data import Dataset
import torch
import pickle
import numpy as np
from .build import DATASETS
from utils.logger import *



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class Neurons(Dataset):
    def __init__(self, config):
        # from ~/Point-MAE/cfgs/dataset_configs/Neurons.yaml
        self.data_root = config.DATA_PATH       # path where there is information about dataset (split)
        self.pc_path = config.PC_PATH           # path where point clouds are stored
        self.labelfile = config.LABEL_FILE
        self.labeltype = config.LABEL_TYPE
        # from ~/Point-MAE/cfgs/pretrain.yaml
        self.sample_points_num = config.npoints # number of downsampled point cloud points
        self.split = config.subset
        
        # HYPERPARAMS NOT IN CONFIG
        self.process_data = False
        self.uniform = False


        self.shape_ids = np.load(os.path.join(self.data_root, f'{self.split}.npy'), allow_pickle=True)

        labels = self._get_labels()
        self.labels = dict(zip(self.shape_ids, labels))

        if self.uniform:
            self.save_path = os.path.join(self.pc_path, 'neurons_%s_%d_pts_fps.dat' % (self.split, self.sample_points_num))
        else:
            self.save_path = os.path.join(self.pc_path, 'neurons_%s_%d_pts.dat' % (self.split, self.sample_points_num))

        with open(os.path.join(self.pc_path, 'idsplit2soma.pkl'), 'rb') as f:
            self.soma_dict = pickle.load(f)
        with open(os.path.join(self.pc_path, 'normalizing_constants.pkl'), 'rb') as f:
            self.normalizing_constants = pickle.load(f)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.shape_ids)
                self.list_of_labels = [None] * len(self.shape_ids)

                for index in tqdm(range(len(self.shape_ids)), total=len(self.shape_ids)):
                    neuron_segment_split = self.shape_ids[index]
                    cls = self.labels[neuron_segment_split]
                    label = np.array([cls])
                    point_set = self._preprocess_shape(neuron_segment_split).astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.sample_points_num)
                    else:
                        point_set = self.random_sample(point_set, self.sample_points_num)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = label

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'Neurons')
        print_log(f'[DATASET] {len(self.shape_ids)} instances were loaded', logger = 'Neurons')

    def _get_label_df(self, df, label, segment_split):
        try:
            label = df[df['segment_split'] == segment_split][label].item()
        except ValueError:
            label = 'None'
        return label
    
    def _get_labels(self):
        import pandas as pd

        df = pd.read_pickle(join(self.pc_path, self.labelfile))
        labels = np.array([self._get_label_df(df, self.labeltype, segment_split) for segment_split in self.shape_ids])
        label2int = {
            '23P':0,
            '4P':1,
            '5P-IT':2,
            '5P-NP':3,
            '5P-PT':4,
            '6P-CT':5,
            '6P-IT':6,
            'BC':7,
            'BPC':8,
            'MC':9,
            'NGC':10,
            'None':11
        }
        labels_int = np.array([label2int[label] for label in labels])
        return labels_int
        # return labels

    def _preprocess_shape(self, neuron_segment_split, normalization_per_shape=False, centering='soma'):
        """ Load, center and normalize pointcloud

        Parameters
        ----------
        neuron_segment_split : str
            segment id of neuron
        pointcloud_path : str
            path where pointclouds are saved
        normalization_per_shape : bool
            normalize the pointcloud per shape or over the whole dataset
        centering : str
            information on what to center the pointcloud; options are 'mean' and 'soma' and 'soma_xz'

        Returns
        -------
        dict
            number of points in pointcloud, ndarray containing 3D points of pointcloud, ndarry containing surface normals of pointcloud
        """
        # multiply with 1000
        coords = np.load(os.path.join(self.pc_path, 'coords', f'{neuron_segment_split}.npz'))['data']*1000

        # Reshape point cloud such that it lies in bounding box of (-1, 1)
        coords = self._center_coords(coords, center_type=centering, neuron_id=neuron_segment_split)
        coords = self._norm_coords(coords, center_type=centering, per_shape=normalization_per_shape)

        return coords

    def _center_coords(self, coords, center_type='soma', **kwargs):
        """ Center the coordinates of the pointcloud to the 1) mean, 2) soma or 3) soma while preserving depth information

        Parameters
        ----------
        coords : ndarray N x 3
            3D coordinates of pointcloud
        center_type : str, optional
            information on what to center the pointcloud, by default 'mean'; other options are 'soma' and 'soma_xz'

        Returns
        -------
        ndarray N x 3
            centered pointcloud
        """
        if center_type == 'mean':
            return coords - np.mean(coords, axis=0, keepdims=True)
        else:
            neuron_id = str(kwargs.get('neuron_id'))
            if center_type == 'soma':
                return coords - self.soma_dict[neuron_id]
            elif center_type == 'soma_xz':
                soma = self.soma_dict[neuron_id]
                soma_xz = np.array([soma[0], 0, soma[2]])
                return coords - soma_xz


    def _norm_coords(self, coords, center_type='soma', per_shape=True):
        """ Normalize coordinates of pointcloud

        Parameters
        ----------
        coords : ndarray N x 3
            3D coordinates of pointcloud
        center_type : str, optional
            information on what to center the pointcloud, by default 'mean'; other options are 'soma' and 'soma_xz'
            influences the range of coordinates for normalization
        per_shape : bool, optional
            normalize the pointcloud per shape or over the whole dataset, by default True
        scale_only : bool, optional
            normalize the pointcloud by scaling or shifting, by default False

        Returns
        -------
        ndarray N x 3
            3D coordinates of normalized pointcloud
        """
        if per_shape:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = self.normalizing_constants[center_type]['max']
            coord_min = self.normalizing_constants[center_type]['min']

        # soma/mean are shifted to optimally use the full volume
        coords = (coords - coord_min) / (coord_max - coord_min)
        coords -= 0.5
        coords *= 2.
        # rotate y-axis
        coords = coords * -1
        return coords

    def random_sample(self, pc, num):
        permutation = np.arange(len(pc))
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

    def __getitem__(self, idx):
        neuron_segment_split = self.shape_ids[idx]
        if self.process_data:
            point_set, label = self.list_of_points[idx], self.list_of_labels[idx]
        else:
            cls = self.labels[neuron_segment_split]
            label = np.array([cls])
            point_set = self._preprocess_shape(neuron_segment_split).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.sample_points_num)
            else:
                point_set = self.random_sample(point_set, self.sample_points_num)
        data = torch.from_numpy(point_set).float()

        # return label[0], neuron_segment_split, (data, label)
        return 'Neuron', neuron_segment_split, (data, label[0])

    def __len__(self):
        return len(self.shape_ids)

""" datasets """
import os

import numpy as np
import torch
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset

from . import globset
from . import mesh
from .. import so3
from .. import se3


class ModelNet(globset.Globset):
    """ [Princeton ModelNet](https://modelnet.cs.princeton.edu/) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class Voltages(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class Atrial(Dataset):
    """ atrial dataset """

    def __init__(self, dataset_path, training=True, transform=None):
        # loader = mesh.offread
        self.dataset_path = dataset_path
        self.transform = transform
        self.training = training
        self.all_examples, self.dirs = self.get_all_examples(dataset_path)
        labels_df = pd.read_csv(f'{dataset_path}/label.csv')
        self.filtered_df = labels_df[labels_df['Study number'].isin(self.dirs)]  # total 8 samples
        # self.get_n_points()  # only need once
        self.template_id = 3  # select i as the template for inference
        if training:
            self.study_ids = self.filtered_df['Study number'].values[:]
            self.af_labels = self.filtered_df['AF type'].values[:]
            self.re_af_labels = self.filtered_df['1Y re AF'].values[:]
        else:
            self.study_ids = np.delete(self.filtered_df['Study number'].values, self.template_id)
            self.af_labels = np.delete(self.filtered_df['AF type'].values, self.template_id)
            self.re_af_labels = np.delete(self.filtered_df['1Y re AF'].values, self.template_id)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        path = f'{self.dataset_path}/Cleaned_PatientData/{study_id}/{study_id}_eam_data.csv'
        df = pd.read_csv(path)
        df = df.sample(n=406, replace=False, random_state=np.random.randint(444))
        points = torch.from_numpy(np.float32(df[['x_norm', 'y_norm', 'z_norm']].values))

        unipolar = torch.from_numpy(df['unipolar'].values)
        bipolar = torch.from_numpy(df['bipolar'].values)

        af_type = torch.from_numpy(np.asarray(self.af_labels[idx]))
        re_af_type = torch.from_numpy(np.asarray(self.re_af_labels[idx]))
        return points, unipolar, bipolar, af_type, re_af_type

    def get_n_points(self):
        n_points = []
        for study_id in self.filtered_df['Study number'].values:
            path = f'{self.dataset_path}/Cleaned_PatientData/{study_id}/{study_id}_eam_data.csv'
            df = pd.read_csv(path)
            n_points.append(len(df))
        print(n_points)  # [765, 406, 1374, 594, 4471, 2683, 1593, 2494]

    def get_template(self):
        study_id = self.filtered_df['Study number'].values[self.template_id]
        path = f'{self.dataset_path}/Cleaned_PatientData/{study_id}/{study_id}_eam_data.csv'
        df = pd.read_csv(path)
        df = df.sample(n=406, replace=False, random_state=np.random.randint(444))
        points = torch.from_numpy(
            np.float32(df[['x_norm', 'y_norm', 'z_norm']].values))

        unipolar = torch.from_numpy(df['unipolar'].values)
        bipolar = torch.from_numpy(df['bipolar'].values)

        af_type = torch.from_numpy(np.asarray(self.filtered_df['AF type'].values[0]))
        re_af_type = torch.from_numpy(np.asarray(self.filtered_df['1Y re AF'].values[0]))
        return points, unipolar, bipolar, af_type, re_af_type

    def __len__(self):
        return len(self.study_ids)

    def get_all_examples(self, dataset_path):
        all_dirs = os.listdir(f'{dataset_path}/Cleaned_PatientData')
        all_example_paths = [f'{dataset_path}/Cleaned_PatientData/{d}/{d}_eam_data.csv' for d in all_dirs]
        return all_example_paths, all_dirs


class AtrialTransform(Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None,
                 template_modifier=None, training=True):
        self.training = training
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, unipolar, bipolar, af_type, re_af_type = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt
        # https://en.wikipedia.org/wiki/Rigid_transformation
        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm
        # p0 = se3.transform(self.rigid_transform.gt.unsqueeze(0), p1) or
        # p1 = se3.transform(self.rigid_transform.igt.unsqueeze(0), p0)

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        if self.training:
            return p0, p1, igt, unipolar, bipolar, af_type, re_af_type
        else:
            template_all = self.dataset.get_template()
            return (p0, unipolar, bipolar, af_type, re_af_type), template_all, p1, igt  # p0 tuple is source in inference


class ShapeNet2(globset.Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """

    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None,
                 template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation(Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self, dataset, perturbation, source_modifier=None,
                 template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = np.array(perturbation)  # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans  # twist or (rotation and translation)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0)  # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0)  # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R  # rotation
            g[:, 0:3, 3] = q  # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(
            np.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt

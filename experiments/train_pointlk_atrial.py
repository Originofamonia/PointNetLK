"""
Example for training a tracker (PointNet-LK).
No-noise version.
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', type=str,
                        default='/home/qiyuan/2021fall/PointNetLK/outputs/atrial',
                        help='output filename (prefix)')  # the result: ${BASENAME}_model_best.pth
    parser.add_argument('-i', '--dataset_path', type=str,
                        default='/home/qiyuan/2021fall/PointNetLK/6363-Project',
                        help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', type=str,
                        # default='/home/qiyuan/2021fall/PointNetLK/6363-Project/label.csv',
                        help='path to the categories to be trained')  # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset_type', default='atrial',
                        choices=['modelnet', 'shapenet2', 'atrial'],
                        metavar='DATASET',
                        help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int, metavar='N',
                        help='points in point-cloud (default: 1024)')
    parser.add_argument('--mag', default=8, type=float, metavar='T',
                        help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str,
                        choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--transfer_from', default='', type=str,
                        metavar='PATH', help='path to pointnet features file')
    parser.add_argument('--dim_k', default=1024, type=int, metavar='K',
                        help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for LK
    parser.add_argument('--max_iter', default=16, type=int,
                        metavar='N', help='max-iter on LK. (default: 10)')
    parser.add_argument('--delta', default=1.0e-2, type=float, metavar='D',
                        help='step size for approx. Jacobian (default: 1.0e-2)')
    parser.add_argument('--learn_delta', dest='learn_delta',
                        default=False,
                        help='flag for training step size delta')

    # settings for on training
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME',
                        help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=1, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD',
                        help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH',
                        help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args(argv)
    return args


def train_ptlk(args, trainset, testset, action):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    LOGGER.debug('Trainer (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    outfile = args.outfile
    suffix = 'best'
    ckpt = torch.load(f'{outfile}_{suffix}.pt')
    model.load_state_dict(ckpt)
    # training
    LOGGER.debug('train, begin')
    # for epoch in range(args.start_epoch, args.epochs):
        # scheduler.step()

        # running_loss, running_info = action.train_1(model, trainloader,
        #                                             optimizer)

        # val_loss, val_info = action.eval_1(model, trainloader)

        # is_best = val_loss < min_loss
        # min_loss = min(val_loss, min_loss)
        #
        # LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss,
        #             val_loss, running_info, val_info)

        # if is_best:
            # snap = {'epoch': epoch + 1,
            #         'model': model.state_dict(),
            #         'min_loss': min_loss,
            #         'optimizer': optimizer.state_dict(), }
            # save_checkpoint(snap, args.outfile, 'snap_best')
    # save_checkpoint(model.state_dict(), args.outfile, 'best')

        # save_checkpoint(snap, args.outfile, 'snap_last')
        # save_checkpoint(model.state_dict(), args.outfile, 'model_last')

    LOGGER.debug('train, end')

    action.infer_plot(model, testset)


def save_checkpoint(state, outfile, suffix):
    torch.save(state, f'{outfile}_{suffix}.pt')


class Action:
    def __init__(self, args):
        # PointNet
        self.args = args
        # self.pointnet = args.pointnet  # tune or fixed
        # self.transfer_from = args.transfer_from
        # self.dim_k = args.dim_k
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg
        # LK
        # self.delta = args.delta
        # self.learn_delta = args.learn_delta
        # self.max_iter = args.max_iter
        self.xtol = 1.0e-7  # t_i in paper
        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        ptnet = ptlk.pointnet.PointNetFeatures(self.args.dim_k, use_tnet=False,
                                               sym_fn=self.sym_fn)
        if self.args.transfer_from and os.path.isfile(self.args.transfer_from):
            ptnet.load_state_dict(
                torch.load(self.args.transfer_from, map_location='cpu'))
        # if self.pointnet == 'tune':
        #     pass
        # el
        if self.args.pointnet == 'fixed':
            for param in ptnet.parameters():
                param.requires_grad_(False)
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.args.delta, self.args.learn_delta)

    def train_1(self, model, trainloader, optimizer):
        model.train()
        vloss = 0.0
        gloss = 0.0
        count = 0
        for i, data in enumerate(trainloader):
            loss, loss_g = self.compute_loss(model, data)

            # forward + backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vloss1 = loss.item()
            vloss += vloss1
            gloss1 = loss_g.item()
            gloss += gloss1
            count += 1

        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count
        return ave_vloss, ave_gloss

    def eval_1(self, model, testloader):
        model.eval()
        vloss = 0.0
        gloss = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                loss, loss_g = self.compute_loss(model, data)

                vloss1 = loss.item()
                vloss += vloss1
                gloss1 = loss_g.item()
                gloss += gloss1
                count += 1

        ave_vloss = float(vloss) / count
        ave_gloss = float(gloss) / count
        return ave_vloss, ave_gloss

    def compute_loss(self, model, data):
        p0, p1, igt, unipolar, bipolar, af_type, re_af_type = data
        p0 = p0.to(self.args.device)  # template
        p1 = p1.to(self.args.device)  # source
        igt = igt.to(self.args.device)  # igt: p0 -> p1
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.args.max_iter,
                                            self.xtol,
                                            self.p0_zero_mean,
                                            self.p1_zero_mean)

        g_est = model.g  # [b, 4, 4], p1 -> p0
        # if epoch == args.epochs - 1:
        #     self.plot_pointcloud(g_est[0], p0[0], p1[0])

        loss_g = ptlk.pointlk.PointLK.comp(g_est, igt)
        # rotated_p1_4 = self.transform(g_est, p1[0])
        # print(rotated_p1_4[:, 0:3] - p0[0])  # correct in train_1

        if self._loss_type == 0:
            loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r
        elif self._loss_type == 1:
            loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r + loss_g
        elif self._loss_type == 2:
            pr = model.prev_r
            if pr is not None:
                loss_r = ptlk.pointlk.PointLK.rsq(r - pr)
            else:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r + loss_g
        else:
            loss = loss_g

        return loss, loss_g

    def infer_plot(self, model, testset):
        """
        infer on training & test sets to plot point cloud
        """
        if not torch.cuda.is_available():
            self.args.device = 'cpu'
        self.args.device = torch.device(self.args.device)

        LOGGER.debug('Trainer (PID=%d), %s', os.getpid(),)

        # model = action.create_model()
        # if self.args.pretrained:
        #     assert os.path.isfile(self.args.pretrained)
        #     model.load_state_dict(
        #         torch.load(self.args.pretrained, map_location='cpu'))
        # model.to(self.args.device)

        checkpoint = None
        if self.args.resume:
            assert os.path.isfile(self.args.resume)
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])

        # dataloader
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=1, shuffle=False, num_workers=self.args.workers)

        # training
        LOGGER.debug('eval, begin')
        model.eval()

        for i, data in enumerate(testloader):
            source_all, template_all = data
            source_all = tuple(t.to(self.args.device) for t in source_all)
            template_all = tuple(t.to(self.args.device) for t in template_all)
            p0, unipolar0, bipolar0, af_type0, re_af_type0 = template_all
            p1, unipolar1, bipolar1, af_type1, re_af_type1 = source_all
            # p0 = p0.to(self.args.device)  # template
            # p1 = p1.to(self.args.device)  # source
            # igt = igt.to(self.args.device)  # igt: p0 -> p1
            r = ptlk.pointlk.PointLK.do_forward(model, p0, p1,
                                                self.args.max_iter,
                                                self.xtol,
                                                self.p0_zero_mean,
                                                self.p1_zero_mean)

            g_est = model.g  # p1 -> p0
            desc = f'before_{i}'
            self.plot_pointcloud(p0[0], p1[0], desc=desc)  # plot before transform
            p1_4 = torch.cat((p1[0], unipolar1[0].unsqueeze(dim=-1)), dim=-1).float()
            print(p1_4.size(), g_est.size())
            rotated_p1_4 = self.transform(g_est, p1[0])
            print(rotated_p1_4[:, 0:3] - p0[0])
            print(p1[0] - p0[0])
            desc = f'after_{i}'
            self.plot_pointcloud(p0[0], rotated_p1_4, desc=desc)

        LOGGER.debug('eval, end')

    def transform(self, g, a):
        """
        g : SE(3),  bs x 4 x 4
        a : R^3,    bs x N x 3
        """
        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
        p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
        if len(g.size()) == len(a.size()):
            b = R.matmul(a) + p.unsqueeze(-1)
        else:
            b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p

        # b = g.matmul(a.unsqueeze(-1)).squeeze(-1)
        return b

    def plot_pointcloud(self, p0, p1, desc):

        # p0_4 = torch.zeros(len(p0)).unsqueeze(1).to(p1)
        # p0_cat = torch.cat((p0, p0_4), dim=-1)
        p1 = p1.detach().cpu().numpy()
        # p0_rotated = torch.matmul(p0_cat, est_g)
        p0 = p0.detach().cpu().numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c='b', label='source')
        ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], c='r', label='template')
        ax.legend()
        plt.savefig(f'{desc}.jpg')


def get_datasets(args):
    cinfo = None
    transform = torchvision.transforms.Compose([
        ptlk.data.transforms.Mesh2Points(),
        ptlk.data.transforms.OnUnitCube(),
        ptlk.data.transforms.Resampler(args.num_points),
    ])

    traindata = ptlk.data.datasets.Atrial(args.dataset_path, training=True,
                                          transform=transform)
    testdata = ptlk.data.datasets.Atrial(args.dataset_path, training=False,
                                         transform=transform)

    mag_randomly = True
    trainset = ptlk.data.datasets.AtrialTransform(traindata,
                                                  ptlk.data.transforms.RandomTransformSE3(
                                                      args.mag,
                                                      mag_randomly), training=True)
    testset = ptlk.data.datasets.AtrialTransform(testdata,
                                                 ptlk.data.transforms.RandomTransformSE3(
                                                     args.mag,
                                                     mag_randomly), training=False)

    return trainset, testset


def main():
    args = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=args.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), args)

    # dataset
    trainset, testset = get_datasets(args)

    # training
    act = Action(args)
    train_ptlk(args, trainset, testset, act)
    args.resume = True
    LOGGER.debug('done (PID=%d)', os.getpid())


if __name__ == '__main__':
    main()

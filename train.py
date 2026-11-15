import argparse
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from network.DSFNet import DSFNet
from network.cd_tools import dice_loss_with_logits, mask_to_edge
from utils import dataloader
from utils.metrics import Evaluator
from utils.tools import AvgMeter, adjust_lr

warnings.filterwarnings('ignore')


def train_one_epoch(train_loader, model, optimizer, epoch, evaluator, bce_loss):
    model.train()
    loss_record = AvgMeter()
    total_step = len(train_loader)
    print('DSF-Net Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))

    for i, sample in enumerate(tqdm(train_loader), start=1):
        optimizer.zero_grad()

        A, B, mask = sample['A'], sample['B'], sample['label']
        A = A.to(device, non_blocking=True)
        B = B.to(device, non_blocking=True)
        Y = mask.to(device, non_blocking=True)
        gts = Y.unsqueeze(1)

        pred, edge_out = model(A, B, gts)
        edge_gt = mask_to_edge(gts)

        loss_cd = bce_loss(pred, gts)
        loss_edge = bce_loss(edge_out, edge_gt) + dice_loss_with_logits(edge_out, edge_gt)
        total_loss = loss_cd + opt.edge_weight * loss_edge

        total_loss.backward()
        optimizer.step()
        loss_record.update(total_loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, '
                  'Edge Loss: {:.4f}, CD Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), loss_edge.item(), loss_cd.item()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, '
                         'Edge Loss: {:.4f}, CD Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step,
                                loss_record.show(), loss_edge.item(), loss_cd.item()))

        output = pred.sigmoid().data.cpu().numpy().squeeze()
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        target = Y.cpu().numpy()
        evaluator.add_batch(target, output.astype(np.int64))

    IoU = evaluator.Intersection_over_Union()[1]
    F1 = evaluator.F1()[1]
    print('Epoch [{:03d}/{:03d}], \n[Training] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--edge_weight', type=float, default=0.1, help='weight for edge supervision')
    parser.add_argument('--data_name', type=str, default='CFD-LEVIR-CD', help='dataset name')
    parser.add_argument('--segclass', type=int, default=1, help='number of segmentation classes')
    parser.add_argument('--save_path', type=str, default='./train_output/DSF-Net/')
    opt = parser.parse_args()

    save_path = opt.save_path + opt.data_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('DSF-Net Learning Rate: {}'.format(opt.lr))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        print('Using CUDA device(s): {} (visible: {})'.format(gpu_count, visible_devices))
    else:
        print('CUDA is not available, using CPU.')

    # build models
    model = DSFNet(num_classes=opt.segclass).to(device)
    if gpu_count > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # set path
    if opt.data_name == 'CFD-LEVIR-CD':
        opt.train_root = "./data/CFD-LEVIR-CD/train/"
        opt.val_root = "./data/CFD-LEVIR-CD/test/"
        palette = [[0, 0, 0], [255, 255, 255]]
    elif opt.data_name == 'CFD-LEVIR-CD+':
        opt.train_root = "./data/CFD-LEVIR-CD+/train/"
        opt.val_root = "./data/CFD-LEVIR-CD+/test/"
        palette = [[0, 0, 0], [255, 255, 255]]
    elif opt.data_name == 'CFD-SYSU-CD':
        opt.train_root = "./data/CFD-SYSU-CD/train/"
        opt.val_root = "./data/CFD-SYSU-CD/val/"
        palette = [[0, 0, 0], [255, 255, 255]]
    elif opt.data_name == 'HH':
        opt.train_root = "./data/HH/train/"
        opt.val_root = "./data/HH/val/"
        palette = [[0, 0, 0], [255, 255, 255]]
    else:
        raise ValueError('Unknown dataset: {}'.format(opt.data_name))

    train_loader = dataloader.get_loader(img_A_root = opt.train_root + 'A/', img_B_root = opt.train_root + 'B/', gt_root = opt.train_root + 'label/', trainsize = opt.trainsize, palette = palette, mode ='train', batchsize = opt.batchsize, mosaic_ratio=0.25, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = dataloader.get_loader(img_A_root = opt.val_root + 'A/', img_B_root = opt.val_root + 'B/', gt_root = opt.val_root + 'label/', trainsize = opt.trainsize, palette = palette, mode ='val', batchsize = opt.batchsize, mosaic_ratio=0, num_workers=4, shuffle=False, pin_memory=True)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("DSF-Net-Train")
    logging.info("Config")
    logging.info('epoch:{}; lr:{}; batchsize:{}; trainsize:{}; edge_weight:{}; save_path:{}'.
                 format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.edge_weight, save_path))

    # loss function
    bce_loss = torch.nn.BCEWithLogitsLoss().to(device)

    evaluator = Evaluator(2)
    for epoch in range(1, (opt.epoch + 1)):
        evaluator.reset()
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train_one_epoch(train_loader, model, optimizer, epoch, evaluator, bce_loss)

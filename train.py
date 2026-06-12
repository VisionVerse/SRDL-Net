import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from datetime import datetime
from utils import dataloader
from utils.metrics import Evaluator
from utils.tools import adjust_lr, AvgMeter
import argparse
import logging
from network.CDRNet import CDRNet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def save_model_state(model, path):
    torch.save(unwrap_model(model).state_dict(), path)

# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    return l2_reg

# ----------------------------------------------------------------------------------------------------------------------

def mask_to_edge(mask):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    edge = (edge > 1e-6).float()
    return edge

def dice_loss_with_logits(logits, targets):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


def Train(train_loader, BCD_Model, BCD_Model_optimizer, epoch, Eva):
    BCD_Model.train()
    loss_record_cdrnet = AvgMeter()
    print('CDRNet Learning Rate: {}'.format(BCD_Model_optimizer.param_groups[0]['lr']))
    for i, sample in enumerate(tqdm(train_loader), start=1):
        BCD_Model_optimizer.zero_grad()
        A, B, mask = sample['A'], sample['B'], sample['label']
        A = Variable(A).to(device, non_blocking=True)
        B = Variable(B).to(device, non_blocking=True)
        Y = Variable(mask).to(device, non_blocking=True)
        gts = Y.unsqueeze(1)

        # train CDRNet
        p_m_prior, p_edge = BCD_Model(A, B, gts)
        bcd_module = unwrap_model(BCD_Model)
        reg_loss = l2_regularisation(bcd_module.scc_1) + l2_regularisation(bcd_module.scc_2) + \
                   l2_regularisation(bcd_module.scc_3) + l2_regularisation(bcd_module.scc_4) + \
                   l2_regularisation(bcd_module.edge_head) + l2_regularisation(bcd_module.decoder)
        reg_loss = opt.reg_weight * reg_loss
        edge_gt = mask_to_edge(gts)

        loss_cd = BCE_loss(p_m_prior, gts)
        loss_edge = BCE_loss(p_edge, edge_gt) + dice_loss_with_logits(p_edge, edge_gt)
        seg_loss = loss_cd + opt.edge_weight * loss_edge + reg_loss

        seg_loss.backward()
        BCD_Model_optimizer.step()

        loss_record_cdrnet.update(seg_loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Edge Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_cdrnet.show(), loss_edge.item()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Edge Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss_record_cdrnet.show(), loss_edge.item()))

        output = p_m_prior.sigmoid().data.cpu().numpy().squeeze()
        output[output>=0.5] = 1
        output[output<0.5] = 0
        target = Y.cpu().numpy()
        # Add batch sample into evaluator
        # print(target.shape, output.shape)
        Eva.add_batch(target, output.astype(np.int64))
    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]
    print('Epoch [{:03d}/{:03d}], \n[Training] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr_cdrnet', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
    parser.add_argument('--lat_weight', type=float, default=1.0, help='weight for latent loss')
    parser.add_argument('--vae_loss_weight', type=float, default=2, help='weight for vae loss')
    parser.add_argument('--reg_weight', type=float, default=0.0, help='weight for regularization term')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for AdamW')
    parser.add_argument('--edge_weight', type=float, default=0.1, help='weight for edge supervision')
    parser.add_argument('--data_name', type=str, default='LEVIR',
                        help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=1,
                        help='')
    parser.add_argument('--save_path', type=str,
                            default='./train_output/CDRNet/')
    opt = parser.parse_args()

    save_path = opt.save_path + opt.data_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('CDRNet Learning Rate: {}'.format(opt.lr_cdrnet))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        print('Using CUDA device(s): {} (visible: {})'.format(gpu_count, visible_devices))
    else:
        print('CUDA is not available, using CPU.')

    # build models
    BCD_Model = CDRNet(latent_dim=opt.latent_dim, num_classes=opt.segclass).to(device)
    if gpu_count > 1:
        BCD_Model = nn.DataParallel(BCD_Model)
    BCD_Model_params = BCD_Model.parameters()
    BCD_Model_optimizer = torch.optim.AdamW(BCD_Model_params, opt.lr_cdrnet, weight_decay=opt.weight_decay)

    # set path
    if opt.data_name == 'LEVIR-CD-HTC136_v10':
        opt.train_root = "/home/xxliu/code/MambaCD/changedetection/data/LEVIR-CD-HTC/train/"
        opt.val_root = "/home/xxliu/code/MambaCD/changedetection/data/LEVIR-CD-HTC/test/"
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'LEVIR-CD+':
        opt.train_root = './Data/Change_Detection/data/LEVIR-CD+/train/'
        opt.val_root = './Data/Change_Detection/data/LEVIR-CD+/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'SYSU':
        opt.train_root = './Data/Change_Detection/data/SYSU-CD/train/'
        opt.val_root = './Data/Change_Detection/data/SYSU-CD/val/'
        palatte = [[0,0,0], [255,255,255]]


    train_loader = dataloader.get_loader(img_A_root = opt.train_root + 'T1/', img_B_root = opt.train_root + 'T2/', gt_root = opt.train_root + 'GT/', trainsize = opt.trainsize, palatte = palatte, mode ='train', batchsize = opt.batchsize, mosaic_ratio=0.25, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = dataloader.get_loader(img_A_root = opt.val_root + 'T1/', img_B_root = opt.val_root + 'T2/', gt_root = opt.val_root + 'GT/', trainsize = opt.trainsize, palatte = palatte, mode ='val', batchsize = opt.batchsize, mosaic_ratio=0, num_workers=4, shuffle=False, pin_memory=True)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("CDRNet-Train")
    logging.info("Config")
    logging.info('epoch:{}; lr_cdrnet:{}; weight_decay:{}; batchsize:{}; trainsize:{}; save_path:{}\
                lat_weight:{} vae_loss_weight: {} reg_loss_weight:{}'.
                format(opt.epoch, opt.lr_cdrnet, opt.weight_decay, opt.batchsize, opt.trainsize, save_path, opt.lat_weight,\
                        opt.vae_loss_weight, opt.reg_weight))

    # loss function
    BCE_loss = torch.nn.BCEWithLogitsLoss().to(device)
    print("Let's go!")
    best_f1 = 0
    best_epoch = 0
    Eva_tr = Evaluator(2)
    Eva_val = Evaluator(2)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        Eva_val.reset()
        cdrnet_lr = adjust_lr(BCD_Model_optimizer, opt.lr_cdrnet, epoch, 0.1, opt.decay_epoch)
        Train(train_loader, BCD_Model, BCD_Model_optimizer, epoch, Eva_tr)
        




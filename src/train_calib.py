import argparse
import os
import glob
import matplotlib
import sys
sys.path.append('../outsourced_models/segmentation_models.pytorch')
matplotlib.use('Agg')  # Set the backend before importing pyplot
from collections import OrderedDict
from datetime import datetime

import segmentation_models_pytorch as smp

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

import archs
from utils import losses
from utils.metrics import iou_score, pixel_accuracy, calculate_ece
from unet_utils import AverageMeterBatched, AverageSumsMeterBatched, EceMeter, str2bool
from utils.train_dataset import SatteliteTrainDataset
from utils.eval_dataset import SatteliteEvalDataset
from utils.utils import make_one_hot
from utils.calibration_losses import Canonical

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<512>"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (how many sampling cycles)')
    parser.add_argument('--train_batches', default=500, type=int, metavar='N',
                        help='number of total samples we take during one train epoch')
    parser.add_argument('--val_batches', default=500, type=int, metavar='N',
                        help='number of total samples we take during one evaluation epoch')
    parser.add_argument('-b', '--train_batch_size', default=8, type=int,
                        metavar='N', help='train-batch size (default: 16)')
    parser.add_argument('-vb', '--val_batch_size', default=8, type=int,
                        metavar='N', help='validation-batch size (default: 16)')
    
    # storing outputs
    parser.add_argument("-o", "--output-dir", default='../outputs/11_channels/', type=str, required=False)

    # model
    parser.add_argument(
        "--continue_train",
        "-cont",
        default=False,
        type=str2bool
    )
    parser.add_argument('--outarch', '-oa', default='Unet',
                        help='choose which outsourced architecture to be used')
    parser.add_argument('--encoder', '-enc', default='vgg16_bn',
                        help='choose which encoder to be used')
    parser.add_argument('--encoder_weights', '-encw', default='imagenet',
                        help='choose which dataset to be used for pretrained weights')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='CustomUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--nb_filters', nargs='+', default=[16,32,64],type=int)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=11, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=14, type=int,
                        help='number of classes (valid classes only - no clouds or unlabeled)')
    parser.add_argument('--which_channels', '-wc', default='all', choices=['all','rgb','rgnir'])
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='CELoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=8, type=int)
    
    #traindataset
    parser.add_argument("-k", "--kaartbladen", default=list(range(30,31)), nargs="+", type=str)
    parser.add_argument("-y", "--years", default=['2022'], nargs="+", type=str)
    parser.add_argument("-m", "--months", default=['01'], nargs="+", type=str)
    parser.add_argument("-r", "--root-dir", default='../allbands_download', type=str, required=False)
    parser.add_argument(
        "-ps",
        "--patch-size",
        default=256,
        type=int,
        help="Size of train patches.",
    )
    
    #valdataset
    parser.add_argument("-vk", "--vkaartbladen", default=list(range(30,31)), nargs="+", type=str)
    parser.add_argument("-vy", "--vyears", default=['2022'], nargs="+", type=str)
    parser.add_argument("-vm", "--vmonths", default=['01'], nargs="+", type=str)
    parser.add_argument("-vr", "--vroot-dir", default="../allbands_download",type=str, required=False)
    parser.add_argument(
        "-vps",
        "--vpatch-size",
        default=256,
        type=int,
        help="Size of val patches.",
    )

    parser.add_argument('-calib', '--ifcalibration', default=True, type=str2bool, help='if apply calibration to model')
    parser.add_argument('-calf', '--calib_factor', default=8, type=int)

    # Preloading data to speed up data loading at the expense of RAM consumption
    parser.add_argument('-plsf', '--preload_sat_flag', default=False, action="store_true", help='whether to preload satellite images')
    parser.add_argument('-plgf', '--preload_gt_flag', default=True, action="store_true", help='whether to preload ground truth')
    parser.add_argument('-plcf', '--preload_cloud_flag', default=True, action="store_true", help='whether to preload cloud masks')


    config = parser.parse_args()

    return config


def check_args(args):
    kaartbladen = args['kaartbladen']
    years = args['years']
    months = args['months']
    root_dir = args['root_dir']

    valid_kaartbladen = True
    if not len(kaartbladen):
        valid_kaartbladen = False
    for kaartblad in kaartbladen:
        print(kaartblad)
        if kaartblad not in [str(item) for item in range(1, 43)]:
            valid_kaartbladen = False
    if not valid_kaartbladen:
        raise ValueError(f"The provided kaartbladen: {kaartbladen} argument is invalid")

    valid_years = True
    if not len(years):
        valid_years = False
    for year in years:
        if year not in [str(item) for item in range(2010, 2023)]:
            valid_years = False
    if not valid_years:
        raise ValueError(f"The provided years: {years} argument is invalid")

    valid_months = True
    if not len(months):
        valid_months = False
    for month in months:
        if month not in [f"{item:02}" for item in range(1, 13)]:
            valid_months = False
    if not valid_months:
        raise ValueError(
            f"The provided months: {months} argument is invalid, the months must be strings representing strings in 'xy' format. eg. '03'"
        )

    valid_root_dir = True
    if not os.path.exists(root_dir):
        valid_root_dir = False
    if not valid_root_dir:
        raise ValueError(
            f"The provided root directory: {valid_root_dir} argument is invalid"
        )

    return



def train(config, train_loader, model, criterion, optimizer, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Current device: {device}")

    avg_meters = {'loss': AverageMeterBatched(),
                  'iou': AverageMeterBatched(),
                  'acc': AverageSumsMeterBatched(),
                  'ece': EceMeter(),
                  'calib_loss': AverageMeterBatched()}

    model.train()
    # model = model.to(device)
    print("train starts:")
    # since the training data is generated on the go,
    # sample as many times as we need in one epoch, the number of samples in one epoch is user defined
    # and the epoch numbers are separately defined as well
    pbar = tqdm(total=config['train_batches'], position=0, leave=True)
    counter = 0
    for index, sample in train_loader:
        idx = index.tolist()
        batchidx = (idx[-1]+1)/config["train_batch_size"] - 1
        print(len(train_loader))
        print(f"index is : {index}")
        sat = sample['sat']
        gt = sample['gt']
        valid_mask = sample["valid_mask"]
        cloud_mask = sample["cloud_mask"]
        label_mask = sample["label_mask"]

        # vvv HACK vvv
        # Works because we apply valid_mask on both the loss and the metric
        # gt - subtact one from all valid pixel labels
        gt[gt > 0] = gt[gt > 0] - 1
        # ^^^ HACK ^^^
        gt = make_one_hot(gt, config['num_classes'])

        input = sat.to(device)
        target = gt.to(device)
        valid_mask = valid_mask.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss, loss_track = criterion(output, target, mask=valid_mask)
                loss += loss
                loss_track += loss_track
            loss /= len(outputs)
            loss_track /= len(outputs)
            iou = iou_score(outputs[-1], target, mask=valid_mask)
            correct, valid = pixel_accuracy(outputs[-1], target, mask=valid_mask)


        else:
            output = model(input)
            loss, loss_track = criterion(output, target, mask=valid_mask)
            if config['ifcalibration'] == True:
                CalLoss = Canonical(0.01, 1, config['calib_factor'])
                calib_loss, calib_loss_track = CalLoss(output, target, mask=valid_mask)
                # remove if necessary
                calib_loss_track_list = [calib_loss_track]
                loss = loss + calib_loss
            iou = iou_score(output, target, mask=valid_mask)  # shape: bs
            correct, valid = pixel_accuracy(output, target, mask=valid_mask)  # shape: bs
            ece, ece_valid = calculate_ece(output, target, 15, mask=valid_mask)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(list(loss_track))
        avg_meters['iou'].update(list(iou))
        avg_meters['acc'].update(list(correct), list(valid))
        avg_meters['ece'].update(ece, sum(ece_valid))
        if config['ifcalibration'] == True:
            avg_meters['calib_loss'].update(calib_loss_track_list)
        else:
            avg_meters['calib_loss'].update([0])

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].report()),
            ('iou', avg_meters['iou'].report()),
            ('acc', avg_meters['acc'].report()),
            ('ece', avg_meters['ece'].report()),
            ('calib_loss', avg_meters['calib_loss'].report())
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        # release some GPU space
        torch.cuda.empty_cache()
        # sample times
        if counter == config['train_batches']:
            break
        else:
            counter = counter+1
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].report()),
                        ('iou', avg_meters['iou'].report()),
                        ('acc', avg_meters['acc'].report()),
                        ('ece', avg_meters['ece'].report()),
                        ('calib_loss', avg_meters['calib_loss'].report())])


def validate(config, val_loader, model, criterion, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Current device: {device}")

    avg_meters = {'loss': AverageMeterBatched(),
                  'iou': AverageMeterBatched(),
                  'acc': AverageSumsMeterBatched(),
                  'ece': EceMeter(),
                  'calib_loss': AverageMeterBatched()}

    # switch to evaluate mode
    model.eval()
    # model = model.to(device)
    print("validation starts:")
    with torch.no_grad():
        pbar = tqdm(total=config['val_batches'], position=0, leave=True)
        counter = 0
        for sample in val_loader:
            sat = sample['sat']
            gt = sample['gt']
            valid_mask = sample["valid_mask"]
            cloud_mask = sample["cloud_mask"]
            label_mask = sample["label_mask"]

            # vvv HACK vvv
            # Works because we apply valid_mask on both the loss and the metric
            # gt - subtact one from all valid pixel labels
            gt[gt > 0] = gt[gt > 0] - 1
            # ^^^ HACK ^^^

            gt = make_one_hot(gt, config['num_classes'])

            input = sat.to(device)
            target = gt.to(device)
            valid_mask = valid_mask.to(device)

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss, loss_track = criterion(output, target, mask=valid_mask)
                    loss += loss
                    loss_track += loss_track
                loss /= len(outputs)
                loss_track /= len(outputs)
                iou = iou_score(outputs[-1], target, mask=valid_mask)
                correct, valid = pixel_accuracy(outputs[-1], target, mask=valid_mask)

            else:
                output = model(input)
                loss, loss_track = criterion(output, target, mask=valid_mask)
                if config['ifcalibration'] == True:
                    CalLoss = Canonical(0.01, 1, config['calib_factor'])
                    calib_loss, calib_loss_track = CalLoss(output, target, mask=valid_mask)
                    # remove if necessary
                    calib_loss_track_list = [calib_loss_track]
                    loss = loss + calib_loss
                iou = iou_score(output, target, mask=valid_mask)  # shape: bs
                correct, valid = pixel_accuracy(output, target, mask=valid_mask)  # shape: bs
                ece, ece_valid = calculate_ece(output, target, 15, mask=valid_mask)

    
            avg_meters['loss'].update(list(loss_track))
            avg_meters['iou'].update(list(iou))
            avg_meters['acc'].update(list(correct), list(valid))
            avg_meters["ece"].update(ece, sum(ece_valid))
            if config['ifcalibration'] == True:
                avg_meters["calib_loss"].update(calib_loss_track_list)
            else:
                avg_meters["calib_loss"].update([0])



            postfix = OrderedDict([
                ('loss', avg_meters['loss'].report()),
                ('iou', avg_meters['iou'].report()),
                ('acc', avg_meters['acc'].report()),
                ('ece', avg_meters['ece'].report()),
                ('calib_loss', avg_meters['calib_loss'].report())
            ])

            #update after all batches have gone through the model
            pbar.set_postfix(postfix)
            pbar.update(1)
            # release some GPU space
            torch.cuda.empty_cache()
            # sample times
            if counter == config['val_batches']:
                break
            else:
                counter = counter + 1
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].report()),
                        ('iou', avg_meters['iou'].report()),
                        ('acc', avg_meters['acc'].report()),
                        ('ece', avg_meters['ece'].report()),
                        ('calib_loss', avg_meters['calib_loss'].report())])


def main():
    from utils.constants import norm_hi_median as norm_hi
    from utils.constants import norm_lo_median as norm_lo
    
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]()

    # if config['ifcalibration'] == 'Canonical':
    #     criterion = CanonicalCalibrationLoss()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    if config['continue_train'] is True:
        if config['outarch'] is not None:
            architecture = getattr(smp, config['outarch'])
            model = architecture(
                encoder_name=config["encoder"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=config["encoder_weights"],  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=config['input_channels'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=14,  # model output channels (number of classes in your dataset)
            )
            model_dir = f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"
            pths = glob.glob(model_dir + "/*.pth")
            pths.sort(key=os.path.getmtime, reverse=True)
            pth = pths[0]
            checkpoint = torch.load(pth)
            model.load_state_dict(checkpoint["model_state_dict"])
            old_epoch = checkpoint["epoch"]
            old_epoch += 1
            old_loss = checkpoint["loss"]

        elif config['arch'] == 'CustomUNet':
            model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['nb_filters'])
            model_dir = f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"
            pths = glob.glob(model_dir + "/*.pth")
            pths.sort(key=os.path.getmtime, reverse=True)
            pth = pths[0]
            checkpoint = torch.load(pth)
            model.load_state_dict(checkpoint["model_state_dict"])
            old_epoch = checkpoint["epoch"]
            old_epoch += 1
            old_loss = checkpoint["loss"]

        else:
            model = archs.__dict__[config['arch']](config['num_classes'],
                                                   config['input_channels'],
                                                   config['deep_supervision'])
            model_dir = f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"
            pths = glob.glob(model_dir + "/*.pth")
            pths.sort(key=os.path.getmtime, reverse=True)
            pth = pths[0]
            checkpoint = torch.load(pth)
            model.load_state_dict(checkpoint["model_state_dict"])
            old_epoch = checkpoint["epoch"]
            old_epoch += 1
            old_loss = checkpoint["loss"]

    elif config['outarch'] is not None:
        architecture = getattr(smp, config['outarch'])
        model = architecture(
            encoder_name=config['encoder'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['encoder_weights'],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=config['input_channels'],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=14,                      # model output channels (number of classes in your dataset)
         )
        if not os.path.exists(
                f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"):
            os.makedirs(
                f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}")

    elif config['arch'] == 'CustomUNet':
        model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['nb_filters'])
        if not os.path.exists(
                f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"):
            os.makedirs(
                f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}")

    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
        if not os.path.exists(
                f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"):
            os.makedirs(
                f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")
    model = model.to(device)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {trainable_params}")

    params = filter(lambda p: p.requires_grad, model.parameters())


    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
        if config['continue_train'] is True:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        if config['continue_train'] is True:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")
    if config['outarch'] is not None:
        with open(f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/config_{date_time}.yml", 'w') as f:
            yaml.dump(config, f)
    else:
        with open(f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/config_{date_time}.yml", 'w') as f:
            yaml.dump(config, f)

    kaartbladen = config['kaartbladen']
    years = config['years']
    months = config['months']
    root_dir = config['root_dir']
    patch_size = config['patch_size']

    norm_hi = np.array(norm_hi)
    norm_lo = np.array(norm_lo)
    if config['which_channels']=='all' and config['input_channels'] == 11:
        which_channels = [0,1,2,3,5,6,7,8,9,10,11]
    elif config['which_channels'] == 'rgb' and config['input_channels'] == 3:
        which_channels = [0,1,2]
    elif config['which_channels'] == 'rgnir' and config['input_channels'] == 3:
        which_channels = [0,1,3]
    else:
        raise ValueError('check which_channels and input_channels')

    train_dataset = SatteliteTrainDataset(
        root_dir,
        kaartbladen,
        years,
        months,
        patch_size=patch_size,
        norm_hi=None,
        norm_lo=None,
        split="train",
        preload_sat_flag=config['preload_sat_flag'],
        preload_gt_flag=config['preload_gt_flag'],
        preload_cloud_flag=config['preload_cloud_flag'],
        which_channels=which_channels
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['train_batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        )
    # sample = next(iter(train_dataloader))
    
    # sat = sample["sat"]
    # gt = sample["gt"]
    # valid_mask = sample["valid_mask"]
    # cloud_mask = sample["cloud_mask"]
    # label_mask = sample["label_mask"]
    # print(sat.shape, gt.shape, valid_mask.shape, cloud_mask.shape, label_mask.shape)

    ##################################
    vkaartbladen = config['vkaartbladen']
    vyears = config['vyears']
    vmonths = config['vmonths']
    vroot_dir = config['vroot_dir']
    vpatch_size = config['vpatch_size']

    norm_hi = np.array(norm_hi)
    norm_lo = np.array(norm_lo)

    val_dataset = SatteliteEvalDataset(
        vroot_dir,
        vkaartbladen,
        vyears,
        vmonths,
        patch_size=vpatch_size,
        norm_hi=None,
        norm_lo=None,
        split="val",
        preload_sat_flag=config['preload_sat_flag'],
        preload_gt_flag=config['preload_gt_flag'],
        preload_cloud_flag=config['preload_cloud_flag'],
        which_channels=which_channels
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
        )
    # vsample = next(iter(val_dataloader))
    # vsat = vsample["sat"]
    # vgt = vsample["gt"]
    # vvalid_mask = vsample["valid_mask"]
    # print(vsat.shape, vgt.shape, vvalid_mask.shape)
    
    # log = OrderedDict([
    #     ('epoch', []),
    #     ('lr', []),
    #     ('loss', []),
    #     ('iou', []),
    #     ('acc', []),
    #     ('val_loss', []),
    #     ('val_iou', []),
    #     ('val_acc', [])
    # ])

    keys = ['epoch', 'lr', 'loss', 'iou', 'acc', 'ece', 'calib_loss', 'val_loss', 'val_iou', 'val_acc', 'val_ece', 'val_calib_loss']
    log = {key: [] for key in keys}

    best_iou = 0
    best_acc = 0
    trigger = 0

    if config['continue_train'] is True:
        loop_list = range(old_epoch, config['epochs'])
    else:
        loop_list = range(config['epochs'])

    for epoch in loop_list:
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_dataloader, model, criterion, optimizer, device)
        # evaluate on validation set
        val_log = validate(config, val_dataloader, model, criterion, device)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - acc %.4f - ece %.4f - calib_loss %.4f - val_loss %.4f - val_iou %.4f - val_acc %.4f - val_ece %.4f - val_calib_loss %.4f'
              % (train_log['loss'], train_log['iou'], train_log['acc'], train_log['ece'], train_log['calib_loss'], val_log['loss'], val_log['iou'], val_log['acc'], val_log['ece'], val_log['calib_loss']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['acc'].append(train_log['acc'])
        log['ece'].append(train_log['ece'])
        log['calib_loss'].append(train_log['calib_loss'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_acc'].append(val_log['acc'])
        log['val_ece'].append(val_log['ece'])
        log['val_calib_loss'].append(val_log['calib_loss'])



        if config['continue_train'] is True:
            if config['outarch'] is not None:
                model_dir = f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"
                csvs = glob.glob(model_dir + "/*.csv")
                csvs.sort(key=os.path.getmtime, reverse=True)
                csv = csvs[0]
                pd.DataFrame(log).iloc[-1:].to_csv(csv, mode='a', header=False)

                trigger += 1
                
                if val_log["acc"] > best_acc:
                    date_time = now.strftime("%Y%m%d%H%M%S")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": train_log["loss"],
                        },
                        f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_bestvalacc.pth",
                    )
                    best_acc = val_log["acc"]
                    print("=> saved best model for validation accuracy")
                    trigger = 0

                date_time = now.strftime("%Y%m%d%H%M%S")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_log["loss"],
                    },
                    f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_lastepoch.pth",
                )
                
                # early stopping
                if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
                    print("=> early stopping")
                    break
            else:
                model_dir = f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}"
                csvs = glob.glob(model_dir + "/*.csv")
                csvs.sort(key=os.path.getmtime, reverse=True)
                csv = csvs[0]
                pd.DataFrame(log).iloc[-1:].to_csv(csv, mode='a', header=False)

                trigger += 1

                if val_log['acc'] > best_acc:
                    date_time = now.strftime("%Y%m%d%H%M%S")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": train_log["loss"],
                        },
                        f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_bestvalacc.pth",
                    )
                    best_acc = val_log['acc']
                    print("=> saved best model for validation accuracy")
                    trigger = 0

                date_time = now.strftime("%Y%m%d%H%M%S")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_log["loss"],
                    },
                    f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_lastepoch.pth",
                )

                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

        else:

            if config['outarch'] is not None:
                pd.DataFrame(log).to_csv(f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/log_{date_time}.csv")

                trigger += 1

                if val_log['acc'] > best_acc:
                    date_time = now.strftime("%Y%m%d%H%M%S")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": train_log['loss'],
                        },
                        f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_bestvalacc.pth",
                    )

                    #torch.save(model.state_dict(), f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}.pth")
                    best_acc = val_log['acc']
                    print("=> saved best model for validation accuracy")
                    trigger = 0

                date_time = now.strftime("%Y%m%d%H%M%S")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_log["loss"],
                    },
                    f"{config['output_dir']}/models/{config['name']}/arch_{config['outarch']}_enc_{config['encoder']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_lastepoch.pth",
                )

                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

            else:
                pd.DataFrame(log).to_csv(f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/log_{date_time}.csv")

                trigger += 1

                if val_log['acc'] > best_acc:
                    date_time = now.strftime("%Y%m%d%H%M%S")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": train_log["loss"],
                        },
                        f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_bestvalacc.pth"
                    )
                    best_acc = val_log['acc']
                    print("=> saved best model for validation accuracy")
                    trigger = 0

                date_time = now.strftime("%Y%m%d%H%M%S")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_log["loss"],
                    },
                    f"{config['output_dir']}/models/{config['name']}/arch_{config['arch']}_train_{config['train_batches']}x{config['train_batch_size']}_val_{config['val_batches']}x{config['val_batch_size']}/model_{date_time}_lastepoch.pth",
                )

                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

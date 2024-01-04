import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp


def fetch_log(arch, pretrained, batch_number, batch_size, acc_or_loss='loss', log_log=False, encoder='imagenet', saved_loc="/esat/gebo/mli1/pycharmproj/geoinformed_clean/outputs/models/backup/20230911", legacy=False):
    model_dir = saved_loc
    # model_dir = "/esat/gebo/mli1/pycharmproj/geoinformed_clean/outputs/wo_blu_w_nir/models/dsb2018_96_CustomUNet_woDS/"
    train_name = f"arch_{arch}_enc_{encoder}_train_{batch_number}x{batch_size}_val_{batch_number}x{batch_size}"
    log_dir = model_dir + train_name
    print(log_dir)

    csvs = glob.glob(log_dir + "/*.csv")
    csvs.sort(key=os.path.getmtime, reverse=True)
    csv = csvs[0]

    ymls = glob.glob(log_dir + "/*.yml")
    ymls.sort(key=os.path.getmtime, reverse=True)
    yml = ymls[0]

    chkpts = glob.glob(log_dir + "/*.pth")
    chkpts.sort(key=os.path.getmtime, reverse=True)
    chkpt = chkpts[0]

    # load model
    architecture = getattr(smp, arch)
    model = architecture(
        encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=14,  # model output channels (number of classes in your dataset)
    )
    # model.load_state_dict(torch.load(chkpt)['model_state_dict'])
    # if legacy:
    #     model.load_state_dict(torch.load(chkpt))
    # else:
    checkpoint = torch.load(chkpt)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(torch.load(chkpt))
    model.eval()

    # get parameters count
    params = sum(p.numel() for p in model.parameters())
    print(f'parameters count of {arch}_{encoder} is {params}')

    # get loss and accuracy data
    df = pd.read_csv(
        f"{csv}", usecols=["epoch", "val_loss", "val_iou", "val_acc"])

    epoch = df["epoch"]
    loss = df["val_loss"]
    acc = df["val_acc"]
    for i in range(len(acc)):
        if acc[i] > 1:
            if i == 0:
                acc[i] = acc[i + 1]
            else:
                acc[i] = acc[i - 1]
    for i in range(len(loss)):
        if loss[i] > 1:
            if i == 0:
                loss[i] = loss[i + 1]
            else:
                loss[i] = loss[i - 1]
    ema_loss = loss.ewm(com=30).mean()
    ema_acc = acc.ewm(com=30).mean()


    if log_log == True:
        if acc_or_loss == 'acc':
            fig = plt.figure(1)
            acc_curve = plt.plot(np.log(epoch), np.log(ema_acc), label=f"{arch}_{encoder}_{legacy}_acc_log")
            # plt.xlim(0, 10)
            legended = plt.legend()
            return yml, acc_curve, fig

        else:
            fig = plt.figure(1)
            loss_curve = plt.plot(np.log(epoch), np.log(ema_loss), label=f"{arch}_{encoder}_{legacy}_loss_log")
            # plt.xlim(0, 10)
            legended = plt.legend()
            return yml, loss_curve, fig
    else:
        if acc_or_loss == 'acc':
            fig = plt.figure(1)
            if legacy:
                acc_curve = plt.plot(epoch, ema_acc, label=f"{arch}_{encoder}_calibration_imagenet_acc")
            else:
                acc_curve = plt.plot(epoch, ema_acc, label=f"{arch}_{encoder}_w/o_calibration_imagenet_acc")
            # plt.xlim(0, 10)
            legended = plt.legend()
            return yml, acc_curve, fig

        else:
            fig = plt.figure(1)
            loss_curve = plt.plot(epoch, ema_loss, label=f"{arch}_{encoder}_{legacy}_loss")
            # plt.xlim(0, 10)
            legended = plt.legend()
            return yml, loss_curve, fig

# def main():
#     fetch_log(
#         arch="Unet",
#         encoder="timm-regnety_002",
#         pretrained="imagenet",
#         batch_number=100,
#         batch_size=8,
#         acc_or_loss="loss",
#         log_log=False,
#     )
#
# if __name__ == '__main__':
#     main()
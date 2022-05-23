import numpy as np  # 导入numpy模块
from PIL import Image  # 入PIL模块用于读取图片，也可使用opencv
import os

import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae
from skimage import io


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, model):
    if type(img) is not torch.tensor:
        x = torch.tensor(img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.2)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
    im_masked = x * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    return im_paste[0]


if __name__ == "__main__":

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    chkpt_dir = 'mae_visualize_vit_large_ganloss.pth'
    model_mae_gan = prepare_model('mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')
    print('Model loaded.')
    torch.manual_seed(2)

    path = 'val/'  # 数据集路径
    # 循环遍历lfw数据集下的所有子文件
    for name_file in os.listdir(path):
        # 遍历子文件下的所有图片文件
        for img_file in os.listdir(path + name_file):
            # print('val/'+str(name_file)+"/"+img_file)  # 打印当前读取的图片名
            # 以下代码根据需要更改
            img = Image.open(path + name_file + '/' + img_file)  # 读取文件
            img = img.resize((224, 224))
            img = np.array(img) / 255.
            if img.shape != (224, 224, 3):
                continue
            img = img - imagenet_mean
            img = img / imagenet_std
            # img = run_one_image(img, model_mae_gan)
            io.imsave('val/'+str(name_file)+"/"+img_file, img)
        print(img_file)
    print("END")

import os

import torch
from torchvision import transforms

from my_dataset import MyDataSet

from model import DynFilter, DFNet, BMNet
import torch.nn.functional as F

import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from math import ceil
from utils import *
from flow_utils import *
from loss import L1_Charbonnier_loss
from PIL import Image
import skimage.metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    #root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target", "00001")  #  data set path
    root = os.path.join(data_root, "vimeo_interp_test", "vimeo_interp_test", "target_f")  #  data set path
    #root = os.path.join(data_root, "vimeo_triplet", "sequences")
    assert os.path.exists(root), "{} path does not exist.".format(root)\

    roots = [x for x in os.listdir(root)]
    #print(roots)
    #print(root)
    examples = []
    for y in roots:
        root1 = os.path.join(root,y)
        #print(root1)
        temp = [x for x in os.listdir(root1) if os.path.isdir(os.path.join(root1, x))]
        for z in temp:
            examples.append(os.path.join(root1, z))
    train_images_path1 = []
    train_images_path2 = []
    train_images_target = []

    #print(examples)

    for x_path in examples:
        #print(x_path)

        images = [img for img in os.listdir(x_path)]
        i = 0
        for img in images:
            img_path = os.path.join(x_path, img)
            if i == 0:
                train_images_path1.append(img_path)
            elif i == 2:
                train_images_path2.append(img_path)
            else:
                #print(img_path)
                train_images_target.append(img_path)
            i += 1

    print("Done!")

    data_transform = {
        "train": transforms.Compose([transforms.RandomCrop((256,448)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])}


    train_data_set = MyDataSet(images_path1=train_images_path1,
                               images_path2=train_images_path2,
                               images_target=train_images_target,
                               transform=data_transform["train"])
    test_data_set = MyDataSet(images_path1=train_images_path1,
                               images_path2=train_images_path2,
                               images_target=train_images_target,
                               transform=data_transform["val"])

    print(len(train_data_set))


    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=1,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                               batch_size=1,
                                               shuffle=True)

    net = BMNet().to(device)
    dnet =DFNet(32,4,16,6).to(device)
    ReLU = torch.nn.ReLU()
    ReLU.to(device)
    context_layer = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    context_layer.load_state_dict(torch.load('Weights/context_layer.pth'))
    context_layer.to(device)
    filtering = DynFilter()
    filtering.to(device)

    print(context_layer)

    net.load_state_dict(torch.load('Weights/BMNet_weights.pth'))
    dnet.load_state_dict(torch.load('Weights/DFNet_weights.pth'))
    for param in net.parameters():
        param.requires_grad = False
    for param in context_layer.parameters():
        param.requires_grad = False
    for param in dnet.parameters():
        param.requires_grad = False

    # get evaluation metrics PSNR and SSIM for test set
    print('\nTesting...')
    with torch.no_grad():
        psnr = 0
        ssim = 0
        num_samples = len(test_loader)
        start_time = time.time()
        cnt = 0

        test_bar = tqdm(test_loader)
        dnet.eval()
        for step, data in enumerate(test_bar):
            I0, I1, It = data
            I0 = I0.to(device)
            I1 = I1.to(device)
            It = It.to(device)

            F_0_1 = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=0) * 2.0
            F_1_0 = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=1) * (-2.0)
            BM = net(F.interpolate(torch.cat((I0, I1), dim=1), (256, 448), mode='bilinear'), time=0.5)

            C1 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), (-0.5) * F_0_1)   # F_t_0
            C2 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), (1-0.5) * F_0_1)  # F_t_1
            C3 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), (0.5) * F_1_0)  # F_t_0
            C4 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), (0.5-1) * F_1_0)   # F_t_1
            C5 = warp(torch.cat((I0, ReLU(context_layer(I0))), dim=1), BM*(-2*0.5))
            C6 = warp(torch.cat((I1, ReLU(context_layer(I1))), dim=1), BM * 2 * (1-0.5))

            #print(tv(BM*(-2*0.5)))

            input = torch.cat((I0,C1,C2,C3,C4,C5,C6,I1),dim=1)
            DF = F.softmax(dnet(input),dim=1)

            candidates = input[:,3:-3,:,:]

            R = filtering(candidates[:, 0::67, :, :], DF)
            G = filtering(candidates[:, 1::67, :, :], DF)
            B = filtering(candidates[:, 2::67, :, :], DF)

            I2 = torch.cat((R, G, B), dim=1)

            It = It.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))
            I2 = I2.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))


            # calculate PSNR and SSIM
            psnr += skimage.metrics.peak_signal_noise_ratio(It, I2, data_range=1)
            ssim += skimage.metrics.structural_similarity(It, I2, data_range=1, multichannel=True)

            time_now = time.time()
            time_taken = time_now - start_time
            start_time = time_now

            cnt += 1

        psnr /= num_samples
        ssim /= num_samples
        print('Test set PSNR: {}, SSIM: {}'.format(psnr, ssim))




if __name__ == '__main__':
    main()

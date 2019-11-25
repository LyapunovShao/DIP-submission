from __future__ import print_function
import os
import random
import torch
import numpy as np
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from dataprocess import process
from models.model import PSMNet

# test parameters

left_img_path = './left0000.png'
right_img_path = './right0000.png'
load_model = './checkpoint_10.tar'
max_disparity = 192
model = PSMNet(max_disparity)
model = torch.nn.DataParallel(model, device_ids=[0])


def get_disp(imgL, imgR):
    model.eval()
    imgL, imgR = torch.Tensor(imgL), torch.Tensor(imgR)
    with torch.no_grad():
        disp = model(imgL, imgR)
    disp = torch.squeeze(disp)
    return disp


def main():
    print("Load model from saved parameters, this program should run on CPU")
    state_dict = torch.load(load_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])

    print("Begin processing images loaded from {}, {}".format(
        left_img_path, right_img_path))
    left_img = Image.open(left_img_path).convert('RGB')
    right_img = Image.open(right_img_path).convert('RGB')
    w, h = left_img.size
    left_img = left_img.crop((w-960, h-544, w, h))
    right_img = right_img.crop((w-960, h-544, w, h))
    processed = process.get_transform()
    imgL = processed(left_img)
    imgR = processed(right_img)
    imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
    imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
    
    pred_disp = get_disp(imgL, imgR)
    img = (pred_disp*256).int()
    skimage.io.imsave('disparity.png', img)
    print("Disparity graph generated successfully")


if __name__ == '__main__':
    main()

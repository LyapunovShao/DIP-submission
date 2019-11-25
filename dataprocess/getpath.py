import torch
from PIL import Image

import os
import os.path


def findpath(path):
    classes = [d for d in os.listdir(
        path) if os.path.isdir(os.path.join(path, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

    monkaa_path = path + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = path + [x for x in disp if 'monkaa' in x][0]

    monkaa_dir = os.listdir(monkaa_path)

    left_paths = []
    right_paths = []
    disp_paths = []

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                left_paths.append(monkaa_path+'/'+dd+'/left/'+im)
                disp_paths.append(monkaa_disp+'/'+dd +
                                  '/left/'+im.split(".")[0]+'.pfm')

    for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
        if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
            right_paths.append(monkaa_path+'/'+dd+'/right/'+im)

    return left_paths, right_paths, disp_paths

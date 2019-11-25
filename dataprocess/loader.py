import torch
import numpy as np
import random
from PIL import Image
import dataprocess.pfm as pfm
import dataprocess.process as preprocess


class ImageDataLoader(torch.utils.data.Dataset):
    def __init__(self, left_set, right_set, disp_set, is_training):
        self.left_set = left_set
        self.right_set = right_set
        self.disp_set = disp_set
        self.is_training = is_training

    def __len__(self):
        return len(self.left_set)

    def __getitem__(self, index):
        processed = preprocess.get_transform()
        # get the data path from index
        left_path, right_path, disp_path = self.left_set[
            index], self.right_set[index], self.disp_set[index]
        left_img, right_img = Image.open(left_path).convert(
            'RGB'), Image.open(right_path).convert('RGB')
        disp_img, _ = pfm.PFM(disp_path)
        disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)

        
        # process the images
        if self.is_training:
            w, h = left_img.size
            th, tw = 256, 512
            x1, y1 = random.randint(0, w-tw), random.randint(0, h-th)

            # randomly select part of the images
            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            disp_img = disp_img[y1:y1+th, x1:x1+tw]

            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, disp_img
        else:
            w, h = left_img.size
            left_img = left_img.crop((w-960, h-544, w, h))
            right_img = right_img.crop((w-960, h-544, w, h))
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, disp_img

import torch
import torchvision.transforms as transforms
import random

def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None):
    normalize = __imagenet_stats
    input_size = 256
    
    return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    
    return transforms.Compose(t_list)

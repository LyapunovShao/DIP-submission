import torch
import torchvision.transforms as transforms
import random

def get_transform(input_size=None,
                  scale_size=None, normalize=None):
    normalize = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
    input_size = 256
    
    return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)

def scale_crop(input_size, scale_size, normalize):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    
    return transforms.Compose(t_list)

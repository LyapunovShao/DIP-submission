import re
import numpy as np
import sys

def PFM(file):
    file = open(file, 'rb')
    title = file.readline().rstrip()
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', bytes.decode(file.readline()))
    width, height = map(int, dim_match.groups())
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
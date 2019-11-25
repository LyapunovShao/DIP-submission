import cv2
import numpy as np
from PIL import Image

path = './'
png = 'disparity.png'
alpha = 1.6

def main():
    depth = cv2.imread(path+png)
    
    color = cv2.applyColorMap(cv2.convertScaleAbs(
        depth, alpha=alpha), cv2.COLORMAP_HSV)
    image = Image.fromarray(color)
    image.save(path+'converted-'+png)


if __name__ == '__main__':
    main()

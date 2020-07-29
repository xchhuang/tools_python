import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
import glob


def cmp(x):
    x = x.split('/')[-1].split('_')[-1].split('.')[0]

    return int(x)

def convert_png2mp4(imgdir, fps):
    
    import imageio
    
    filename = imgdir + '/a.mp4'
    # print (imgdir, filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    try:
        writer = imageio.get_writer(filename, fps=fps)
    except Exception:
        imageio.plugins.ffmpeg.download()
        writer = imageio.get_writer(filename, fps=fps)

    imgs = sorted(glob.glob("{}/*.png".format(imgdir)))
    imgs.sort(key=cmp)
    # print (imgs)

    for img in imgs:
        fn = img.split('\\')[-1].split('_')
        print (fn)
        im = imageio.imread(img)
        writer.append_data(im)
    writer.close()




def main():
    
    convert_png2mp4('test/out/256_2classes_sample4/video', fps=2)





if __name__ == '__main__':
    main()




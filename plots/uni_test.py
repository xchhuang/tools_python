import numpy as np
import torch
import matplotlib.pyplot as plt
from poisson_disk import generate_possion_dis
from jitter import jitter_sampler
import torch
import glob
from tqdm import tqdm


def read_from_txt(path):
    p = []
    c = []
    num_classes = 12
    id_map = {}
    class_index = [1000 * (i + 1) for i in range(num_classes)]
    for i in range(num_classes):
        id_map[class_index[i]] = i

    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            line = [float(x) for x in line]
            cur_p = [float(x) / 1e4 for x in line[1:]]
            cur_c = id_map[int(line[0])]
            # cur_c = int(line[0])
            # cur_c = int(0)

            p.append(cur_p)
            c.append(cur_c)
    p = np.array(p)
    c = np.array(c)
    # c[0:32] = 0
    # c[32:] = 1
    # print(p.shape)
    return p, c


def plot_point_np(pts, cls, size, title):
    plt.figure(1)
    plt.scatter(pts[:, 0], pts[:, 1], s=size, c=cls)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def vis():
    files = glob.glob('../../pattern-synthesis/data/mix2/train/*.txt')
    for idx, file in enumerate(tqdm(files)):
        # print(file)
        pts, cls = read_from_txt(file)
        # print(pts.shape, cls.shape)
        plot_point_np(pts, cls, 10, 'results/{:}'.format(idx))


def main():
    vis()


if __name__ == '__main__':
    main()

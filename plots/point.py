import numpy as np
import torch
import matplotlib.pyplot as plt
from poisson_disk import generate_possion_dis
from jitter import jitter_sampler
import torch


def normalization(points, d_space=2, edge_space=0.1, egde_feature=0.2, norm=True):
    """
    taken from PPS, to leave some edge_space for point patterns
    """
    min = points.min(0)[0]
    max = points.max(0)[0]
    for id in range(d_space):
        r = max[id] - min[id]
        points[:, id] = (((points[:, id] - min[id]) / r) * (1 - 2 * edge_space) + edge_space)
    return points


def read_from_pt(path):
    p = torch.load(path)
    p = p[:, 0:2]
    normalization(p, edge_space=0.02)
    p = p.detach().cpu().numpy()
    c = np.zeros(p.shape[0])
    # print(p.shape, c.shape)
    # print(p)
    return p, c


def read_from_txt(path):
    p = []
    c = []
    num_classes = 2
    id_map = {}
    class_index = [1000 * (i + 1) for i in range(num_classes)]
    for i in range(num_classes):
        id_map[class_index[i]] = i

    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            line = [float(x) for x in line]
            cur_p = [float(x) for x in line[1:]]
            # cur_c = id_map[int(line[0])]
            cur_c = int(line[0])

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


def gen_point_pattern():
    N = 256
    size = 20
    pts = np.random.rand(N, 2)
    cls = np.zeros(N, dtype=np.int32)
    plot_point_np(pts, cls, size, 'random')

    pts = generate_possion_dis(N, 0.01, 0.99)
    cls = np.zeros(N, dtype=np.int32)
    plot_point_np(pts, cls, size, 'poisson')

    pts = jitter_sampler(N)
    cls = np.zeros(N, dtype=np.int32)
    plot_point_np(pts, cls, size, 'jitter')


def gen_point_pattern2():
    # path = '../../pattern-synthesis/image_based/results/vshape_s_N64_chamfer/optimize_style_loss2/pts_scale2.txt'
    # pts, cls = read_from_txt(path)
    # plot_point_np(pts, cls, 5, 'vshape_s_output')
    # path = '../../pattern-synthesis/data/vshape_s/64/test/00000.txt'
    # pts, cls = read_from_txt(path)
    # plot_point_np(pts, cls, 20, 'vshape_s_input')

    # path = '../../pattern-synthesis/image_based/results/vshape_N64_chamfer/optimize_poisson/pts_scale2.txt'
    # pts, cls = read_from_txt(path)
    # plot_point_np(pts, cls, 5, 'poisson_output')
    # path = '../../pattern-synthesis/data/poisson/64/test/00000.txt'
    # pts, cls = read_from_txt(path)
    # plot_point_np(pts, cls, 20, 'poisson_input')
    pass


def gen_point_pattern3():
    path = '../../PointPatternSynthesis/pytorch_original/data/Building1f.pt'
    pts, cls = read_from_pt(path)
    plot_point_np(pts, cls, 5, 'Building1_input')


def main():
    gen_point_pattern3()


if __name__ == '__main__':
    main()

import numpy as np
import torch
import matplotlib.pyplot as plt
from poisson_disk import generate_possion_dis
from jitter import jitter_sampler


def plot_point_np(pts, cls, size, title):
    plt.figure(1)
    plt.scatter(pts[:, 0], pts[:, 1], s=size, c=cls)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close('all')


def main():
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


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import art3d
import numpy as np
from scipy import spatial

import pyosa


def show_pcd(xyz, p=None, patch=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*xyz.T,
               fc='w', ec='k', s=7, lw=0.5)
    if p is not None:
        ax.plot(*p, 'ro', mec='k', mew=0.5, ms=4, zorder=3)
    if patch:
        ax.add_patch(patch)
        art3d.pathpatch_2d_to_3d(patch, z=p[2], zdir='z')
    ax.set_box_aspect([1, 1, 1]),
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-20, vertical_axis='y')
    return fig, ax


def main():
    # stanford bunny 3d point cloud model
    xyz = np.genfromtxt('bunny100k.xyz', delimiter=' ')
    
    # downsampling the point cloud just a bit for better visualization
    N = xyz.shape[0]
    num = 0.05 * N
    mask = np.arange(0, N, int(N/num))
    xyz_ds = xyz[mask]
    
    # some arbitrary, open surface for which we want to extract the area
    point = xyz_ds[np.argmax(xyz_ds[:, 2]), :]
    tree = spatial.KDTree(xyz_ds)
    r = 0.5
    ind = tree.query_ball_point(point, r)
    
    # show it
    circle = patches.Circle(point, r, fc='none', ec='k')
    fig, ax = show_pcd(xyz_ds, point, circle)
    
    # extraction of the surface is as simple as
    surf = xyz_ds[ind]
    area = pyosa.estimate(surf)
    
    ax.set_title(f'surface area: {area:.2f}')
    fig.savefig('bunny.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()

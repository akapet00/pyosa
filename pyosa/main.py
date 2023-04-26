import logging
logging.basicConfig(level=logging.CRITICAL)

import numpy as np
from scipy import spatial


__all__ = ['estimate']


def _infer_knn(N):
    knn = int(2 * np.log(N))
    if knn < 5:
        knn = 5
    elif knn > 30:
        knn = 30
    return knn


def _change_of_basis(X):
    X = X.copy()
    mu = X.mean(axis=0)
    X = X - mu
    C = X.T @ X  # convariance matrix
    U, _, _ = np.linalg.svd(C)
    return X @ U


def estimate(xyz, n=None, smooth=False, **kwargs):
    """Return estimated surface area of an open surface sampled and
    given as a point cloud.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud distributed in 3-D space of shape (N, 3) where N is
        the total number of points on the surface.
    n : numpy.ndarray, optional
        Oriented normals in 3-D space of shape (N, 3) where N is the
        number of points in a point cloud. If not provided, normals are
        automatically inferred from `xyz` which can lead to numerical
        artifacts depending on the size of the `xyz`.
    smooth : bool, optional
        If set to True, open surface is treated as smooth surface.
        Smoothing filter will be applied to the reconstructed triangle
        mesh without any shrinkage.
    kwargs : dict, optional
        Additional keyword arguments for
        `open3d.geometry.TriangleMesh.create_from_point_cloud_poisson`
        function.
    
    Returns
    -------
    float or tuple
        Estimated surface area and reconstructed mesh (vertices and
        triangles stored in dictionary) if `full_output` is True.
    """
    try:
        import open3d as o3d
    except ImportError:
        print('Missing requirement. Please install `open3d`.')
    else:
        N = xyz.shape[0]
        if N < 10:
            raise ValueError('Number of points must be > 10.')
        # set the point cloud in its orthonormal basis
        xyzt = _change_of_basis(xyz)
        # create the convex hull on tangential plane
        hull = spatial.Delaunay(xyzt[:, :2])
        # create the point cloud instance
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzt)
        if n is not None:
            pcd.normals = o3d.utility.Vector3dVector(n)
        else:
            knn = _infer_knn(N)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
            pcd.orient_normals_consistent_tangent_plane(knn)
        pcd.normalize_normals()  # set the magnitude of each normal to 1
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, **kwargs)
        mesh.remove_duplicated_vertices()  # clean up reconstructed surface
        mesh.remove_degenerate_triangles()  # product of removing vertices
        vert = np.asarray(mesh.vertices)  # export reconstructed vertices
        # remove vertices positioned out of the convex hull
        vert_mask = hull.find_simplex(vert[:, :2]) < 0
        mesh.remove_vertices_by_mask(vert_mask)
        # optional smoothing
        if smooth:
            mesh = mesh.filter_smooth_taubin()
        return mesh.get_surface_area()

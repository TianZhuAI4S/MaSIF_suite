"""Serve unchanged Open3D0.5 registration using safe numeric patch packets (Python3.6)."""

import json
import sys

import numpy as np
import open3d as o3d


def cloud(points, normals, descriptors):
    """Construct original Open3D point cloud and feature matrix in Å coordinates."""
    result = o3d.PointCloud()
    result.points = o3d.Vector3dVector(points)
    result.normals = o3d.Vector3dVector(normals)
    feature = o3d.Feature()
    feature.data = descriptors.T
    return result, feature


def align(packet):
    """Run original RANSAC2000/max_validation500 followed by point-to-plane ICP1Å."""
    with np.load(packet, allow_pickle=False) as values:
        source, source_feature = cloud(values['source_points'], values['source_normals'], values['source_descriptors'])
        target, target_feature = cloud(values['target_points'], values['target_normals'], values['target_descriptors'])
    result = o3d.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature, 1.5,
        o3d.TransformationEstimationPointToPoint(False), 3,
        [o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.CorrespondenceCheckerBasedOnDistance(1.0),
         o3d.CorrespondenceCheckerBasedOnNormal(np.pi / 2)],
        o3d.RANSACConvergenceCriteria(2000, 500))
    result = o3d.registration_icp(source, target, 1.0, result.transformation,
                                o3d.TransformationEstimationPointToPlane())
    return dict(transformation=np.asarray(result.transformation).tolist(), fitness=float(result.fitness))


def main():
    """Read explicit packet requests until EOF; preserve exceptions in the worker log."""
    np.random.seed(42)
    for line in sys.stdin:
        request = json.loads(line)
        print('MASIF_RESULT ' + json.dumps(align(request['packet']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

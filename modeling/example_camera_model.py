import numpy as np
import cv2
from deepsport_utilities.calib import Calib, Point2D, Point3D


def compute_camera(points2d, points3d, output_shape):

    height, width = output_shape
    h = np.eye(3, 4)
    h[2, 3] = 1.0
    calib = Calib.from_P(h, width=width, height=height)
    # cv2 calibrateCamera requires at least 4 points
    if len(points2d) > 5:
        points2d_ = Point2D(np.array(points2d).T)
        points3d_ = Point3D(np.array(points3d).T)
        points2D = points2d_.T.astype(np.float32)
        points3D = points3d_.T.astype(np.float32)

        try:
            _, K, kc, r, t = cv2.calibrateCamera(
                [points3D],
                [points2D],
                (width, height),
                None,
                None,
                None,
                None,
                cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST,
            )
        except cv2.error as err:
            print(err)
            return calib

        T = t[0]
        R = cv2.Rodrigues(r[0])[0]

        try:
            calib = Calib(width=width, height=height, T=T, R=R, K=K, kc=kc)
        except np.linalg.LinAlgError:
            print('no')
            pass
    # knowing that there's no distortion
    # calib = calib.update(kc=None)

    return calib

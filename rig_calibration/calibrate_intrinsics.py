"""
Per-camera intrinsic calibration from checkerboard images.

Run this once per physical camera (camA, camB) using ~15-20 handheld shots
of a checkerboard at varying distance, tilt, and position in the frame.
Straight-on, centered-only shots give a poorly-constrained calibration.

Usage:
    python calibrate_intrinsics.py \
        --images_dir "./calib/camA/*.jpg" \
        --board_size 9 6 \
        --square_size 0.024 \
        --out_json ./calib/camA_intrinsics.json

`--board_size` is the number of INNER corners (cols rows), not squares
(a 10x7-square board has 9x6 inner corners).
`--square_size` is in meters.
"""
import argparse
import glob
import json
import cv2
import numpy as np


def calibrate(image_paths, board_size, square_size):
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    used = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f'skip (unreadable): {path}')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f'checkerboard not found: {path}')
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used += 1

    if used < 8:
        raise RuntimeError(
            f'only {used} usable checkerboard views found (need >= 8-10 for a '
            'stable calibration) -- capture more, varying board tilt/distance/position '
            'in-frame, not just straight-on shots.'
        )

    reproj_err, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print(f'used {used}/{len(image_paths)} images, reprojection error: {reproj_err:.4f} px')
    if reproj_err > 1.0:
        print('WARNING: reprojection error > 1px is high for this kind of calibration -- '
              'double check board_size/square_size and image sharpness before trusting this.')

    dist = dist.flatten()
    return {
        'camera_model': 'OPENCV',
        'w': img_size[0],
        'h': img_size[1],
        'fl_x': float(K[0, 0]),
        'fl_y': float(K[1, 1]),
        'cx': float(K[0, 2]),
        'cy': float(K[1, 2]),
        'k1': float(dist[0]) if len(dist) > 0 else 0.0,
        'k2': float(dist[1]) if len(dist) > 1 else 0.0,
        'p1': float(dist[2]) if len(dist) > 2 else 0.0,
        'p2': float(dist[3]) if len(dist) > 3 else 0.0,
        'k3': float(dist[4]) if len(dist) > 4 else 0.0,
        'reprojection_error_px': float(reproj_err),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images_dir', required=True,
                    help='glob pattern for checkerboard images, e.g. "./calib/camA/*.jpg"')
    p.add_argument('--board_size', type=int, nargs=2, required=True, metavar=('COLS', 'ROWS'),
                    help='number of INNER corners, e.g. 9 6 for a 10x7-square board')
    p.add_argument('--square_size', type=float, required=True,
                    help='checkerboard square size in meters')
    p.add_argument('--out_json', required=True)
    args = p.parse_args()

    image_paths = sorted(glob.glob(args.images_dir))
    if not image_paths:
        raise RuntimeError(f'no images matched {args.images_dir}')

    result = calibrate(image_paths, tuple(args.board_size), args.square_size)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {args.out_json}')

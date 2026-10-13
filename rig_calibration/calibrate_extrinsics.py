"""
Per-position extrinsic ("camera-to-world") calibration via a single
checkerboard image per rig position.

Concept: place ONE checkerboard flat where the plant pot will later sit --
this single, fixed placement defines your world origin/ground plane -- and
photograph it from every rig position you intend to use for real captures
(all 12: both cameras, all angular steps). Each image is solved independently
with solvePnP, but because they all reference the same physical board
placement, all 12 poses land in one consistent world frame.

IMPORTANT: the checkerboard must not move between these 12 shots. If it
moves, every position calibrated after the move is in a different world
frame than the ones before it, and nothing downstream will be consistent --
this fails silently (no error), so double-check before you start shooting.

Because this only depends on the rig's mechanical positions, not on any
particular plant, you calibrate it once and reuse the output for every
specimen afterward -- PROVIDED your rig positions are physically repeatable
(hard stops/detents). If your rig doesn't have repeatable positions, you'll
need to redo this per capture session instead of once for the whole study.

Usage:
    python calibrate_extrinsics.py \
        --manifest ./calib/positions_manifest.json \
        --intrinsics camA=./calib/camA_intrinsics.json camB=./calib/camB_intrinsics.json \
        --board_size 9 6 \
        --square_size 0.024 \
        --out_json ./calib/rig_positions.json \
        --visualize

manifest.json format:
    {"pos00": {"camera": "camA", "image": "./calib/positions/pos00.jpg"},
     "pos01": {"camera": "camB", "image": "./calib/positions/pos01.jpg"},
     ...}
"""
import argparse
import json
import cv2
import numpy as np


def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    K = np.array([
        [d['fl_x'], 0, d['cx']],
        [0, d['fl_y'], d['cy']],
        [0, 0, 1],
    ])
    dist = np.array([d.get('k1', 0), d.get('k2', 0), d.get('p1', 0), d.get('p2', 0), d.get('k3', 0)])
    return K, dist


def solve_position(image_path, K, dist, board_size, square_size):
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f'cannot read {image_path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError(f'checkerboard not found in {image_path}')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f'solvePnP failed for {image_path}')
    rvec, tvec = cv2.solvePnPRefineLM(objp, corners, K, dist, rvec, tvec)

    # Reprojection error as a per-position sanity check.
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean())

    R, _ = cv2.Rodrigues(rvec)
    # world-to-camera, OpenCV convention: +X right, +Y down, +Z forward (into the scene)
    w2c_cv = np.eye(4)
    w2c_cv[:3, :3] = R
    w2c_cv[:3, 3] = tvec.flatten()
    c2w_cv = np.linalg.inv(w2c_cv)

    # OpenCV/COLMAP camera axes (+Y down, +Z forward) -> NeRF/OpenGL camera axes
    # (+Y up, +Z backward) that Nerfstudio's transforms.json expects. Flipping the
    # Y and Z columns of the rotation re-labels the camera's own local axes -- it
    # does not touch the world frame or the translation.
    c2w_nerf = c2w_cv.copy()
    c2w_nerf[:3, 1] *= -1
    c2w_nerf[:3, 2] *= -1

    return c2w_nerf, err


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True)
    p.add_argument('--intrinsics', nargs='+', required=True,
                    help='camera_id=path pairs, e.g. camA=./calib/camA_intrinsics.json camB=...')
    p.add_argument('--board_size', type=int, nargs=2, required=True, metavar=('COLS', 'ROWS'))
    p.add_argument('--square_size', type=float, required=True)
    p.add_argument('--out_json', required=True)
    p.add_argument('--visualize', action='store_true',
                    help='plot solved camera positions/orientations around the world origin')
    args = p.parse_args()

    intrinsics = {}
    for kv in args.intrinsics:
        cam_id, path = kv.split('=', 1)
        intrinsics[cam_id] = load_intrinsics(path)

    with open(args.manifest) as f:
        manifest = json.load(f)

    results = {}
    errs = []
    for pos_id, entry in manifest.items():
        K, dist = intrinsics[entry['camera']]
        c2w, err = solve_position(entry['image'], K, dist, tuple(args.board_size), args.square_size)
        errs.append(err)
        print(f'{pos_id} ({entry["camera"]}): reprojection error {err:.3f} px')
        if err > 1.0:
            print(f'  WARNING: high reprojection error for {pos_id} -- check this image/board detection')
        results[pos_id] = {
            'camera': entry['camera'],
            'transform_matrix': c2w.tolist(),
        }

    print(f'mean reprojection error across {len(errs)} positions: {np.mean(errs):.3f} px')

    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {args.out_json}')

    if args.visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([0], [0], [0], c='k', marker='x', s=80, label='world origin (board)')
        for pos_id, r in results.items():
            c2w = np.array(r['transform_matrix'])
            pos = c2w[:3, 3]
            forward = -c2w[:3, 2]  # camera looks down -Z in NeRF convention
            ax.scatter(*pos, c='C0')
            ax.text(*pos, pos_id, fontsize=8)
            ax.quiver(*pos, *forward, length=0.1, color='C1')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('solved rig positions (arrows = camera forward direction)\nsanity check: arrows should point roughly inward, toward the origin')
        plt.legend()
        plt.show()

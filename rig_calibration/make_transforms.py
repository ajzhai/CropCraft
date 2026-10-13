"""
Assemble a Nerfstudio-format transforms.json for one specimen capture, from:
  - the rig's pre-solved per-position extrinsics (calibrate_extrinsics.py output)
  - per-camera intrinsics (calibrate_intrinsics.py output)
  - a manifest of this specimen's captured RGB/depth files per position

Usage:
    python make_transforms.py \
        --rig_positions ./calib/rig_positions.json \
        --intrinsics camA=./calib/camA_intrinsics.json camB=./calib/camB_intrinsics.json \
        --frames_manifest ./data/plant_007/frames.json \
        --depth_unit_scale 0.001 \
        --out_json ./data/plant_007/transforms.json

`--depth_unit_scale` converts your depth image's stored integer values to
meters (e.g. 0.001 if depth is stored as uint16 millimeters). This depends
entirely on whatever depth registration/export step produced your depth
images -- confirm the actual units before trusting this, don't assume.

frames_manifest.json format (paths relative to --out_json's directory,
matching Nerfstudio convention):
    {"pos00": {"rgb": "images/pos00.png", "depth": "depth/pos00.png"},
     "pos01": {"rgb": "images/pos01.png", "depth": "depth/pos01.png"},
     ...}
`depth` is optional per frame if you want an RGB-only run.
"""
import argparse
import json


def build(rig_positions, intrinsics, frames_manifest, depth_unit_scale):
    frames = []
    for pos_id, frame_entry in frames_manifest.items():
        if pos_id not in rig_positions:
            raise KeyError(f'{pos_id} in frames_manifest has no calibrated rig position')
        pos = rig_positions[pos_id]
        cam_intr = intrinsics[pos['camera']]

        frame = {
            'file_path': frame_entry['rgb'],
            'transform_matrix': pos['transform_matrix'],
            'fl_x': cam_intr['fl_x'],
            'fl_y': cam_intr['fl_y'],
            'cx': cam_intr['cx'],
            'cy': cam_intr['cy'],
            'w': cam_intr['w'],
            'h': cam_intr['h'],
            'k1': cam_intr.get('k1', 0.0),
            'k2': cam_intr.get('k2', 0.0),
            'p1': cam_intr.get('p1', 0.0),
            'p2': cam_intr.get('p2', 0.0),
        }
        if 'depth' in frame_entry:
            frame['depth_file_path'] = frame_entry['depth']
        frames.append(frame)

    out = {
        'camera_model': 'OPENCV',
        'frames': frames,
    }
    if depth_unit_scale is not None:
        out['depth_unit_scale_factor'] = depth_unit_scale
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rig_positions', required=True)
    p.add_argument('--intrinsics', nargs='+', required=True,
                    help='camera_id=path pairs, e.g. camA=./calib/camA_intrinsics.json camB=...')
    p.add_argument('--frames_manifest', required=True)
    p.add_argument('--depth_unit_scale', type=float, default=None)
    p.add_argument('--out_json', required=True)
    args = p.parse_args()

    with open(args.rig_positions) as f:
        rig_positions = json.load(f)

    intrinsics = {}
    for kv in args.intrinsics:
        cam_id, path = kv.split('=', 1)
        with open(path) as f:
            intrinsics[cam_id] = json.load(f)

    with open(args.frames_manifest) as f:
        frames_manifest = json.load(f)

    result = build(rig_positions, intrinsics, frames_manifest, args.depth_unit_scale)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {args.out_json} with {len(result["frames"])} frames')

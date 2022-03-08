from ai2thor.controller import Controller
import h5py
import argparse
import os
from vnenv.environments.ai2thor_env.thordata_utils import get_scene_names
from vnenv.environments.ai2thor_env.agent_pose_state import AgentPoseState
import json
from tqdm import tqdm
from ai2thor.platform import CloudRendering
from collections import defaultdict
scenes = {'kitchen': '25'}
# scenes = {'kitchen': '1-30', 'living_room': '1-30',
#           'bedroom': '1-30', 'bathroom': '1-30'},
v_fn = 'visible_map.json'
t_fn = 'trans.json'
move_list = [0, 1, 1, 1, 0, -1, -1, -1]
horizons = [0, 30]


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--frame", action='store_true')
    parser.add_argument("--depth", action='store_true')
    parser.add_argument("--seg_frame", action='store_true')
    parser.add_argument("--seg_mask", action='store_true')
    parser.add_argument("--detection", action='store_true')
    parser.add_argument("--rotate_angle", type=int, default=45)
    parser.add_argument("--grid_size", type=float, default=0.25)
    parser.add_argument("--vis_dist", type=float, default=1)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    args = parser.parse_args()
    return args


def dump(scene, ctrler, path, obs_key, rotate_angle):
    if not os.path.exists(path):
        os.makedirs(path)
    evt = ctrler.reset(scene=scene)
    # visible data
    objectIDs = [x['objectId'] for x in evt.metadata['objects']]
    visible_data = {}
    for obj in objectIDs:
        poses = ctrler.step(
            action="GetInteractablePoses",
            objectId=obj,
            horizons=horizons
        ).metadata["actionReturn"]
        str_poses = []
        for p in poses:
            p.pop('standing')
            str_poses.append(str(AgentPoseState(**p)))
        visible_data[obj] = str_poses
    with open(os.path.join(path, v_fn), "w") as fp:
        json.dump(visible_data, fp)
    # trans data & obs data
    positions = ctrler.step(
        action="GetReachablePositions"
    ).metadata["actionReturn"]
    trans_data = {}
    seg_map = {}
    h5_writer = {x: h5py.File(os.path.join(path, x+'.hdf5'), 'w')
                 for x in obs_key}
    pbar = tqdm(
        total=len(positions)*len(horizons)*(360//rotate_angle),
        leave=False)
    pos_mark = defaultdict(int)
    for p in positions:
        pos_mark[(p['x'], p['z'])] = 1
    for p in positions:
        for r in range(0, 360, rotate_angle):
            for h in horizons:
                pbar.update(1)
                evt = ctrler.step(
                    action="Teleport",
                    position=p,
                    rotation=dict(x=0, y=r, z=0),
                    horizon=h
                )
                assert evt.metadata['lastActionSuccess']
                key = str(AgentPoseState(
                    p['x'], p['y'], p['z'],
                    rotation=r, horizon=h))
                # obs data
                for x in obs_key:
                    if x == 'seg_frame':
                        h5_writer[x].create_dataset(
                            key, data=evt.instance_segmentation_frame)
                        seg_map[key] = evt.object_id_to_color
                    elif x == 'seg_mask':
                        msub = h5_writer[x].create_group(key)
                        for k, v in evt.instance_masks.items():
                            msub.create_dataset(k, data=v)
                    elif x == 'detection':
                        msub = h5_writer[x].create_group(key)
                        for k, v in evt.instance_detections2D.items():
                            msub.create_dataset(k, data=v)
                    elif x == 'depth':
                        h5_writer[x].create_dataset(key, data=evt.depth_frame)
                    else:
                        h5_writer[x].create_dataset(key, data=evt.frame)
                # trans data
                evt = ctrler.step(action="MoveAhead")
                x = evt.metadata['agent']['position']['x']
                z = evt.metadata['agent']['position']['z']
                trans_data[key] = int(
                    evt.metadata['lastActionSuccess'] and pos_mark[(x, z)])
    pbar.close()
    with open(os.path.join(path, t_fn), "w") as fp:
        json.dump(trans_data, fp)
    if 'seg_frame' in obs_key:
        with open(os.path.join(path, 'seg_map.json'), "w") as fp:
            json.dump(seg_map, fp)
    for v in h5_writer.values():
        v.close()


if __name__ == '__main__':
    args = init_parser()
    path = args.path
    if not os.path.exists(path):
        os.makedirs(path)
    rotate_angle = args.rotate_angle
    assert rotate_angle > 0
    assert 360 % rotate_angle == 0
    obs_key = []
    for x in ['frame', 'seg_frame', 'seg_mask', 'detection', 'depth']:
        if getattr(args, x):
            obs_key.append(x)
    ctrler = Controller(
        width=args.width, height=args.height, renderDepthImage=args.depth,
        renderInstanceSegmentation=(args.seg_mask or args.seg_frame),
        rotateStepDegrees=rotate_angle, gridSize=args.grid_size,
        visibilityDistance=args.vis_dist, platform=CloudRendering)
    for s in tqdm(iterable=get_scene_names(scenes)):
        dump(s, ctrler, os.path.join(path, s), obs_key, rotate_angle)
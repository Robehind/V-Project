from torchvision import transforms as T
import h5py
import os
from tqdm import tqdm
from taskenvs.ai2thor_env.thordata_utils import get_scene_names
from models import my_resnet50, resnet18fm
import torch
import argparse
# 300x300:[0.5265, 0.4560, 0.3682]), 'std': tensor([0.0540, 0.0554, 0.0567]
# 128x128:[0.5269, 0.4565, 0.3687]), 'std': tensor([0.0540, 0.0554, 0.0567]
"""生成resnet特征"""
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='../vdata/thordata/')
parser.add_argument("--resnet50fc", action='store_true')
parser.add_argument("--resnet50score", action='store_true')
parser.add_argument("--resnet18fm", action='store_true')
parser.add_argument("--no_norm", action='store_true')
parser.add_argument("--batch_sz", type=float, default=64)
args = parser.parse_args()
scenes = {
        'kitchen': '1-30',
        'living_room': '1-30',
        'bedroom': '1-30',
        'bathroom': '1-30'}
image_name = 'frame.hdf5'
datadir = args.path
batch_sz = args.batch_sz
norm = T.Normalize(mean=[0.5265, 0.4560, 0.3682],
                   std=[0.0540, 0.0554, 0.0567],
                   inplace=True)
featkey, models = [], []
if args.resnet50fc or args.resnet50score:
    models.append(my_resnet50().cuda())
if args.resnet18fm:
    models.append(resnet18fm().cuda())
for k in ['resnet50fc', 'resnet50score', 'resnet18fm']:
    if getattr(args, k):
        featkey.append(k)
scene_names = get_scene_names(scenes)
print(f'making for {len(scene_names)} scenes')

for scene_name in tqdm(iterable=scene_names):
    scene_path = os.path.join(datadir, scene_name)
    mid = '_nn' if args.no_norm else ''
    writer = {
        k: h5py.File(os.path.join(scene_path, k+mid+'.hdf5'), 'w')
        for k in featkey}
    RGBloader = h5py.File(os.path.join(scene_path, image_name), "r")
    keys, x = [], []
    frames = len(RGBloader.keys())
    pbar = tqdm(total=frames, leave=False)
    for f, k in enumerate(RGBloader.keys()):
        keys.append(k)
        tmp = T.ToTensor()(RGBloader[k][:])
        x.append(tmp)
        if len(keys) == batch_sz or f == frames-1:
            x = torch.stack(x)
            if not args.no_norm:
                x = norm(x)
            x = x.cuda()
            outs = {}
            for m in models:
                outs.update(m(x))
            for i, sk in enumerate(keys):
                for k in writer:
                    writer[k].create_dataset(sk, data=outs[k][i])
            pbar.update(len(keys))
            keys, x = [], []
    pbar.close()
    RGBloader.close()
    for k in writer:
        writer[k].close()

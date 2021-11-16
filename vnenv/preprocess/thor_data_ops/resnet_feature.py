from torchvision import transforms as T
import h5py
import os
from tqdm import tqdm
from utils import get_scene_names, states_num
from models import my_resnet50
import torch
# 300x300:[0.5265, 0.4560, 0.3682]), 'std': tensor([0.0540, 0.0554, 0.0567]
# 128x128:[0.5269, 0.4565, 0.3687]), 'std': tensor([0.0540, 0.0554, 0.0567]
"""生成新的fc和fc score文件"""

scenes = {
        'kitchen': '25',
        # 'living_room': '1-30',
        # 'bedroom': '1-30',
        # 'bathroom': '1-30',
    }
image_name = 'images.hdf5'
datadir = '../vdata/thordata/'
fc_name = 'resnet50fc_no_norm.hdf5'
score_name = 'resnet50score.hdf5'
batch_sz = 64
norm = T.Normalize(mean=[0.5265, 0.4560, 0.3682],
                   std=[0.0540, 0.0554, 0.0567],
                   inplace=True)
scene_names = get_scene_names(scenes)
resmodel = my_resnet50().cuda()
print(f'making for {len(scene_names)} scenes')

pbar = tqdm(total=states_num(scenes, datadir, preload=image_name))
for scene_name in scene_names:
    scene_path = os.path.join(datadir, scene_name)

    fc_writer = h5py.File(os.path.join(scene_path, fc_name), 'w')
    # score_writer = h5py.File(os.path.join(scene_path, score_name), 'w')
    RGBloader = h5py.File(os.path.join(scene_path, image_name), "r",)
    keys, x = [], []
    frames = len(RGBloader.keys())
    for f, k in enumerate(RGBloader.keys()):
        keys.append(k)
        tmp = T.ToTensor()(RGBloader[k][:])
        x.append(tmp)

        if len(keys) == batch_sz or f == frames-1:
            x = torch.stack(x)
            # x = norm(x)
            x = x.cuda()
            out = resmodel(x)

            resnet_fc = out['fc'].cpu().numpy()
            # resnet_s = out['s'].cpu().numpy()
            # print(resnet_score.shape)
            for i, sk in enumerate(keys):
                fc_writer.create_dataset(sk, data=resnet_fc[i])
                # score_writer.create_dataset(sk, data=resnet_s[i])
            pbar.update(len(keys))
            keys = []
            x = []
    RGBloader.close()
    fc_writer.close()
    # score_writer.close()

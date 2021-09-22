import torchvision.models as models
from torchvision import transforms as T
import torch.nn as nn
import torch
import h5py
import os
from tqdm import tqdm
import random
from total_states import states_num, get_scene_names, make_scene_name 
from mean_std import get_mean_std
from models import my_resnet50
"""生成新的fc和fc score文件"""

scenes = {
        'kitchen':range(1,21),
        'living_room':range(1,21),
        'bedroom':range(1,21),
        'bathroom':range(1,21),
    }
image_name = 'images.hdf5'
datadir = '../mixed_offline_data/'
fc_name = 'resnet50fc.hdf5'
score_name = 'resnet50score'

trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5265, 0.4560, 0.3682],std=[0.0540, 0.0554, 0.0567],inplace=True)
])
scene_names = get_scene_names(scenes)
resmodel = my_resnet50().cuda()
print(f'making for {len(scene_names)} scenes')
    
pbar = tqdm(total = states_num(scenes, preload=image_name))
for scene_name in scene_names:

    fc_writer = h5py.File(os.path.join(datadir,scene_name,fc_name), 'w')
    score_writer = h5py.File(os.path.join(datadir,scene_name,score_name), 'w')
    RGBloader = h5py.File(os.path.join(datadir,scene_name,image_name),"r",)
    
    for k in RGBloader.keys():
        pbar.update(1)
        x = RGBloader[k][:]
        x = trans(x).unsqueeze(0)
        x = x.cuda()
        out = resmodel(x)

        resnet_fc = out['fc']
        resnet_s = out['s']
        #print(resnet_score.shape)
        fc_writer.create_dataset(k, data = resnet_fc.cpu().numpy())
        score_writer.create_dataset(k, data = resnet_s.cpu().numpy())
        #print(resnet_fc.shape)
        #break
    RGBloader.close()
    fc_writer.close()
    score_writer.close()
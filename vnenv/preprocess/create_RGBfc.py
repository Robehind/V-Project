import torchvision.models as models
from torchvision import transforms as T
import torch.nn as nn
import torch
import h5py
import os
from tqdm import tqdm
from total_states import states_num, get_scene_names, make_scene_name 
import random
from models import Encoder1
"""生成RGBpred的fc文件"""

test_scenes = {
        'kitchen':range(16,21),
        'living_room':range(16,21),
        'bedroom':range(16,21),
        'bathroom':range(16,21),
    }
load_model_dir = 'trained_models/rgbencode1/enc_215714.dat'
out_name = 'rgbpred_fc_nc.hdf5'

Normalize = T.Normalize(
    mean=[0.5269, 0.4565, 0.3687], 
    std=[0.0540, 0.0554, 0.0567], 
    inplace=True
    )
trans = T.Compose([
    T.ToTensor(),
    #Normalize
])
model = Encoder1().cuda()
model.load_state_dict(torch.load(load_model_dir))
scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')
    
pbar = tqdm(total = states_num(test_scenes, preload='images128.hdf5'))
for n in scene_names:
    datadir = '../mixed_offline_data/'

    fc_writer = h5py.File(os.path.join(datadir,n,out_name), 'w')
    RGBloader = h5py.File(os.path.join(datadir,n,'images128.hdf5'), "r",)
    
    for k in RGBloader.keys():
        pbar.update(1)
        pic = RGBloader[k][:]
        data =trans(pic).unsqueeze(0).cuda()
        out = model(data).detach()
        out = torch.flatten(out)
        fc_writer.create_dataset(k, data = out.cpu().numpy())
    RGBloader.close()
    fc_writer.close()
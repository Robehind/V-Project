import h5py
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from total_states import states_num, get_scene_names, make_scene_name 

datadir = '../mixed_offline_data/'
data_name = 'images.hdf5'
new_data_name = 'normal_tensor128.hdf5'
test_scenes = {
        'kitchen':range(1,31),
        'living_room':range(1,31),
        'bedroom':range(1,31),
        'bathroom':range(1,31),
    }

scene_names = get_scene_names(test_scenes)

trans = T.Compose([
    T.ToPILImage(),
    T.Resize((128,128)),
    #T.ToTensor(),
    #T.Normalize(mean=[0.5114, 0.4435, 0.3569],std=[0.0537, 0.0553, 0.0566],inplace=True)
])

pbar = tqdm(total = states_num(test_scenes, preload=data_name))
for s in scene_names:
    data = {}

    loader = h5py.File(os.path.join(datadir,s,data_name),"r",)
    num = len(list(loader.keys()))
    for k in loader.keys():
        vobs = trans(loader[k][:]).unsqueeze(0)

        data[k] = vobs.numpy()
        #data[k] = loader[k][:].squeeze()
    #print(num)
    
    loader.close()
    writer = h5py.File(os.path.join(datadir,s,new_data_name),"w",)
    for k in data:
        writer.create_dataset(k, data = data[k])
        pbar.update(1)
    writer.close()
    
    

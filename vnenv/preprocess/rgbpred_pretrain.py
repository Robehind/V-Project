import h5py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from total_states import get_scene_names
import random
import numpy as np
from tensorboardX import SummaryWriter
from models import EncoderT as ENC
from models import DecoderT as DEC

# ######################training##################################
datadir = '../mixed_offline_data/'
data_name = 'images128.hdf5'
test_scenes = {
        'kitchen': range(1, 16),
        'living_room': range(1, 16),
        'bedroom': range(1, 16),
        'bathroom': range(1, 16),
    }
path_to_save = './trained_models/rgbencodeT2'
print_freq = 10000
save_freq = 2e6
batch_size = 64
total_frames = 3e7

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
scene_names = get_scene_names(test_scenes)

enc = ENC().cuda()
dec = DEC().cuda()

model = nn.Sequential(
    enc,
    dec,
)

optim = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

log_writer = SummaryWriter(log_dir=path_to_save)

n_frames = 0
print_gate_frames = print_freq
save_gate_frames = save_freq
loss_record = 0
count = 0
Normalize = T.Normalize(
    mean=[0.5269, 0.4565, 0.3687],
    std=[0.0540, 0.0554, 0.0567],
    inplace=True
    )
trans = T.Compose([
    T.ToTensor(),
    # Normalize
])
pbar = tqdm(total=total_frames)
while 1:

    random.shuffle(scene_names)
    for s in scene_names:

        loader = h5py.File(os.path.join(datadir, s, data_name), "r",)
        keys = list(loader.keys())
        random.shuffle(keys)
        num = len(keys)
        runs = num // batch_size
        batch_keys = [keys[i*batch_size:(i+1)*batch_size] for i in range(runs)]
        for i in range(runs):
            # ####输入128，128，3的255图像时
            data = torch.stack([trans(loader[x][:]) for x in batch_keys[i]])
            data = data.cuda()
            # label = torch.stack([T.ToTensor()(loader[x][:])
            # for x in batch_keys[i]])
            # label = label.cuda()

            out = model(data)
            loss = F.l1_loss(out, data.detach(), reduction='sum')
            loss_record += loss.cpu().item()/batch_size
            count += 1
            loss.backward()
            optim.step()
            model.zero_grad()

            n_frames += batch_size
            pbar.update(batch_size)

            if n_frames >= print_gate_frames:
                print_gate_frames += print_freq
                log_writer.add_scalar("loss", loss_record/count, n_frames)
                loss_record = 0
                count = 0

            if n_frames >= save_gate_frames:
                save_gate_frames += save_freq
                enc_to_save = enc.state_dict()
                all_to_save = model.state_dict()
                import time
                start_time = time.time()
                time_str = time.strftime(
                    "%H%M%S", time.localtime(start_time)
                )
                save_path = os.path.join(
                    path_to_save,
                    f"enc_{time_str}.dat"
                )
                torch.save(enc_to_save, save_path)
                save_path = os.path.join(
                    path_to_save,
                    f"model_{time_str}.dat"
                )
                torch.save(all_to_save, save_path)
            if n_frames >= total_frames:
                loader.close()
                exit()
        loader.close()

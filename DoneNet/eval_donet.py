from collections import defaultdict
import h5py
import os
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import numpy as np
import json
from taskenvs.ai2thor_env.utils import get_scene_names
from done_model import DoneNet
from settings import datadir, wd_path, data_name, targets
import math
vscenes = {'kitchen': '1,4,20,21,30',
           'living_room': '1,12,14,17,27',
           'bedroom': '3,6,14,17,27',
           'bathroom': '3,4,7,24,27'}
s, t = 0.5, 0.0001
base = 0.5
num = int(math.log(t/s, base))
mula = s / base
model_path = './PriorDoneNet/DoneNet_4000.dat'
model = DoneNet(2048+300, 512, 0).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()
# prepare wd data
tgt_num = len(targets)
wd_loader = h5py.File(wd_path, "r")
wd_data = []
for t in targets:
    wd_data.append(wd_loader['fasttext'][t][:])
wd_tensor = torch.tensor(np.array(wd_data))
wd_loader.close()
# prepare label data
v_scenes = get_scene_names(vscenes)
vis_data = {s: defaultdict(lambda: [0]*tgt_num) for s in v_scenes}

TPRs, FPRs = [], []
for p in tqdm(iterable=np.logspace(1, num, num=num, base=base)):
    pred_gate = mula*p
    for s in v_scenes:
        with open(os.path.join(datadir, s, 'visible_map.json'), 'r') as f:
            vis_json = json.load(f)
        for k in vis_json:
            tgt = k.split("|")[0]
            if tgt in targets:
                for pos in set(vis_json[k]):
                    vis_data[s][pos][targets.index(tgt)] = 1
    # scene loaders
    loader = {s: h5py.File(os.path.join(datadir, s, data_name))
              for s in v_scenes}
    pose_keys = {s: list(loader[s].keys()) for s in v_scenes}
    val_nums = 0
    for s in v_scenes:
        val_nums += len(pose_keys[s])
    # evaluating
    pbar = tqdm(total=val_nums, leave=False,
                desc=f'validating {pred_gate}')
    TPR, FPR = 0, 0
    TPR_c, FPR_c = 0, 0
    for s in v_scenes:
        sps, lbs = [], []
        poses = pose_keys[s]
        sps.append(torch.stack(
            [torch.tensor(loader[s][x][:]) for x in poses]))
        lbs.append(torch.stack(
            [torch.tensor(vis_data[s][x]) for x in poses]))
        sps = torch.stack(sps, dim=1).reshape(-1, 2048)
        lbs = torch.stack(lbs, dim=1).reshape(-1, tgt_num).float()
        for j in range(tgt_num):
            data = torch.cat(
                [sps, wd_tensor[j].repeat(len(poses), 1)], dim=1)
            label = lbs[:, j].unsqueeze(1).cuda()
            with torch.no_grad():
                out = model(data.cuda())
            Tprob = out[label == 1].cpu()
            TPR += (Tprob >= pred_gate).sum().item()
            TPR_c += Tprob.shape[0]
            Fprob = out[label == 0].cpu()
            FPR += (Fprob >= pred_gate).sum().item()
            FPR_c += Fprob.shape[0]
        pbar.update(len(poses))
    pbar.close()
    TPR /= TPR_c
    FPR /= FPR_c
    TPRs.append(TPR)
    FPRs.append(FPR)
plt.scatter(FPRs, TPRs)
for i, p in enumerate(np.logspace(1, num, num=num, base=base)):
    plt.annotate(
        str(mula*p), xy=(FPRs[i], TPRs[i]), xytext=(FPRs[i]+0.01, TPRs[i]))
plt.show()

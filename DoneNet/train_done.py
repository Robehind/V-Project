from collections import defaultdict
import h5py
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
import numpy as np
import json
from taskenvs.ai2thor_env.utils import get_scene_names
import wandb
from done_model import DoneNet
from settings import datadir, wd_path, data_name, tscenes, vscenes, targets
# ######################training##################################
args = dict(
    epoch=5000,
    batch_scenes=10,
    scene_samples=64,
    lr=0.0007,
    model='DoneNet',
    pred_gate=0.2,
    dprate=0,
    decay=0
)
debug = False
decay = args['decay']
dprate = args['dprate']
epoch = args['epoch']
pred_gate = args['pred_gate']
batch_scenes = args['batch_scenes']
scene_samples = args['scene_samples']
val_ep_freq = 100
batch_sz = batch_scenes * scene_samples
exp_name = 'PriorDoneNet'
save_path = './'+exp_name
########################################################
model = DoneNet(2048+300, 512, dprate).cuda()
model.train()
optim = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=decay)
if not debug:
    wandb.init(project='DoneTrain', entity='robehind', name=exp_name,
               config=args, dir='.')
########################################################
# prepare wd data
tgt_num = len(targets)
wd_loader = h5py.File(wd_path, "r")
wd_data = []
for t in targets:
    wd_data.append(wd_loader['fasttext'][t][:])
wd_tensor = torch.tensor(np.array(wd_data))
wd_loader.close()
# prepare label data
t_scenes = get_scene_names(tscenes)
v_scenes = get_scene_names(vscenes)
vis_data = {s: defaultdict(lambda: [0]*tgt_num) for s in t_scenes+v_scenes}
for s in t_scenes+v_scenes:
    with open(os.path.join(datadir, s, 'visible_map.json'), 'r') as f:
        vis_json = json.load(f)
    for k in vis_json:
        tgt = k.split("|")[0]
        if tgt in targets:
            for pos in set(vis_json[k]):
                kk = pos.split('|')
                kk[2] = str(int(kk[2]) % 360)
                _pos = '|'.join(kk)
                vis_data[s][_pos][targets.index(tgt)] = 1
# scene loaders
loader = {s: h5py.File(os.path.join(datadir, s, data_name))
          for s in t_scenes+v_scenes}
pose_keys = {s: list(loader[s].keys())
             for s in t_scenes+v_scenes}
val_nums = 0
for s in v_scenes:
    val_nums += len(pose_keys[s])
# train
for ep in tqdm(iterable=range(epoch)):
    record = {'loss': 0, 'TPR': 0, 'FPR': 0}
    count = 0
    random.shuffle(t_scenes)
    pbar = tqdm(total=len(t_scenes)*scene_samples, leave=False)
    for i in range(0, len(t_scenes), batch_scenes):
        neg_sps, pos_sps = [], []
        sps, lbs = [], []
        for s in t_scenes[i:i+batch_scenes]:
            pos_sps = random.sample(
                list(vis_data[s].keys()), scene_samples//4)
            neg_sps = random.sample(pose_keys[s], scene_samples*3//4)
            poses = pos_sps + neg_sps
            random.shuffle(poses)
            sps.append(torch.stack(
                [torch.tensor(loader[s][x][:]) for x in poses]))
            lbs.append(torch.stack(
                [torch.tensor(vis_data[s][x]) for x in poses]))
        sps = torch.stack(sps, dim=1).reshape(-1, 2048)
        lbs = torch.stack(lbs, dim=1).reshape(-1, tgt_num).float()
        loss, TPR, FPR = 0, 0, 0
        TPR_c, FPR_c = 0, 0
        for roll in range(batch_sz-tgt_num+1):
            data = torch.cat([sps[roll:roll+tgt_num], wd_tensor], dim=1)
            label = lbs[roll:roll+tgt_num].diag().unsqueeze(1).cuda()
            out = model(data.cuda())
            loss += F.binary_cross_entropy(out, label)
            Tprob = out[label == 1].cpu()
            TPR += (Tprob >= pred_gate).sum().item()
            TPR_c += Tprob.shape[0]
            Fprob = out[label == 0].cpu()
            FPR += (Fprob >= pred_gate).sum().item()
            FPR_c += Fprob.shape[0]
        TPR /= TPR_c
        FPR /= FPR_c
        # loss /= batch_sz-tgt_num+1
        loss.backward()
        optim.step()
        model.zero_grad()
        record['loss'] += loss.cpu().item()
        record['TPR'] += TPR
        record['FPR'] += FPR
        count += 1
        pbar.update(batch_sz)
    pbar.close()
    # validate
    if (ep+1) % val_ep_freq == 0:
        # save model
        title = model.__class__.__name__ + '_' + str(ep+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pp = os.path.join(save_path, title+".dat")
        state_to_save = model.state_dict()
        torch.save(state_to_save, pp)

        # pbar = tqdm(total=val_nums, leave=False,
        #             desc='validating')
        # model.eval()
        # TPR, FPR = 0, 0
        # TPR_c, FPR_c = 0, 0
        # for s in v_scenes:
        #     sps, lbs = [], []
        #     poses = pose_keys[s]
        #     sps.append(torch.stack(
        #         [torch.tensor(loader[s][x][:]) for x in poses]))
        #     lbs.append(torch.stack(
        #         [torch.tensor(vis_data[s][x]) for x in poses]))
        #     sps = torch.stack(sps, dim=1).reshape(-1, 2048)
        #     lbs = torch.stack(lbs, dim=1).reshape(-1, tgt_num).float()
        #     for j in range(tgt_num):
        #         data = torch.cat(
        #             [sps, wd_tensor[j].repeat(len(poses), 1)], dim=1)
        #         label = lbs[:, j].unsqueeze(1).cuda()
        #         with torch.no_grad():
        #             out = model(data.cuda())
        #         Tprob = out[label == 1].cpu()
        #         TPR += (Tprob >= pred_gate).sum().item()
        #         TPR_c += Tprob.shape[0]
        #         Fprob = out[label == 0].cpu()
        #         FPR += (Fprob >= pred_gate).sum().item()
        #         FPR_c += Fprob.shape[0]
        #     pbar.update(len(poses))
        # pbar.close()
        # model.train()
        # TPR /= TPR_c
        # FPR /= FPR_c
        # if not debug:
        #     wandb.log({'val/TPR': TPR, 'val/FPR': FPR}, commit=False)
    if not debug:
        wandb.log({'train/'+k: v/count for k, v in record.items()})

for v in loader.values():
    v.close()

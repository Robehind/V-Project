from collections import defaultdict
from typing import Dict
# 'scene' 'target' 'success' 'model' 'return'
# 'actions' 'poses' 'events' 'agent_done' 'min_acts
metrics1 = ['SR', 'SPL', 'ER', 'CR', 'VC', 'steps', 'Done']
metrics2 = ['Et', 'Ct', 'ERt', 'CRt']


def measure_epi(epi: Dict):
    # SPL
    if epi['success']:
        SPL = epi['min_acts'] / len(epi['actions'])
    else:
        SPL = 0
    assert SPL <= 1
    # Et, ERt
    Et, ERt = Explore(epi)
    # Ct, CRt
    Ct, CRt = Collision(epi)
    epidata = dict(
        scene=epi['scene'], target=epi['target'], model=epi['model'],
        min_acts=epi['min_acts'], SR=epi['success'],
        Et=Et, ERt=ERt, Ct=Ct, CRt=CRt, steps=len(epi['actions']),
        SPL=SPL, ER=ERt[-1], CR=CRt[-1], poses=epi['poses'],
        Done=int(epi['agent_done']), actions=epi['actions'],
        VC=epi['vis_cnt'])
    return epidata


def Explore(epi: Dict):
    # 计算Et和ERt
    marker = defaultdict(int)
    poses = epi['poses']
    marker[poses[0]] = 1
    Et, ERt = [], []
    Es = 0.0
    for t, p in enumerate(poses[1:]):
        Et.append(marker[p])
        Es += marker[p]
        ERt.append(Es / (t+1))
        marker[p] = 1
    return Et, ERt


def Collision(epi: Dict):
    # 计算Ct和CRt
    events = epi['events']
    Ct, CRt = [], []
    Cs = 0.0
    for t, e in enumerate(events):
        Ct.append(int(e == 'collision'))
        Cs += float(e == 'collision')
        CRt.append(Cs / (t+1))
    return Ct, CRt

# def CalcSPLt(epi: Dict, minActsCache: Dict):
#     epi_length = len(epi['actions'])
#     scene = epi['scene']
#     target = epi['target']
#     if epi['success']:
#         SPLt = []
#         for i, p in enumerate(epi['poses'][:-1]):
#             min_acts = minActsCache[scene][target][p] + 1
#             SPLt.append(min_acts / float(epi_length - i))
#     else:
#         SPLt = [0.0] * epi_length
#     return SPLt

from tqdm import tqdm
import h5py
import os


def get_scene_names(scenes):
    """根据参数生成完整的房间的名字"""
    tmp = []
    for k in scenes.keys():
        ranges = [x for x in scenes[k].split(',')]
        number = []
        for a in ranges:
            ss = [int(x) for x in a.split('-')]
            number += range(ss[0], ss[-1]+1)
        number = list(set(number))
        tmp += [make_scene_name(k, i) for i in number]
    return tmp


def make_scene_name(scene_type, num):
    mapping = {"kitchen": '', "living_room": '2',
               "bedroom": '3', "bathroom": '4'}
    front = mapping[scene_type]
    endd = '_physics' if (front == '' or front == '2') else ''
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num) + endd
    return "FloorPlan" + front + "0" + str(num) + endd


def states_num(scenes, datadir, preload):

    scene_names = get_scene_names(scenes)
    count = 0
    pbar = tqdm(total=len(scene_names), desc='Gathering...', leave=False)
    for s in scene_names:
        RGBloader = h5py.File(os.path.join(datadir, s, preload), "r",)
        num = len(list(RGBloader.keys()))
        count += num
        pbar.update(1)
        RGBloader.close()
    return count

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


def get_type(scene_name):
    """根据房间名称返回该房间属于哪个类型"""
    mapping = {'2': 'living_room', '3': 'bedroom', '4': 'bathroom'}
    num = scene_name.split('_')[0].split('n')[-1]
    if len(num) < 3:
        return 'kitchen'
    return mapping[num[0]]


def make_scene_name(scene_type, num):
    """根据房间的类别和序号生成房间的名称
    例如，scene_type = kitchen的第num = 5个房间，为FloorPlan5
    """
    mapping = {"kitchen": '', "living_room": '2',
               "bedroom": '3', "bathroom": '4'}
    front = mapping[scene_type]
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num)
    return "FloorPlan" + front + "0" + str(num)


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

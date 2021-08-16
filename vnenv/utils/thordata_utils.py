import random
import os

def get_scene_names(train_scenes):
    """根据参数生成完整的房间的名字"""
    tmp = {}
    for k in train_scenes.keys():
        ranges = [x for x in train_scenes[k].split(',')]
        number = []
        for a in ranges:
            ss = [int(x) for x in a.split('-')]
            number += range(ss[0], ss[-1]+1)
        number = list(set(number))
        tmp[k] = [make_scene_name(k, i) for i in number]
    return tmp

def get_type(scene_name):
    """根据房间名称返回该房间属于哪个类型"""
    mapping = {'2':'living_room','3':'bedroom', '4':'bathroom'}
    num = scene_name.split('_')[0].split('n')[-1]
    if len(num) < 3:
        return 'kitchen'
    return mapping[num[0]]

def make_scene_name(scene_type, num):
    """根据房间的类别和序号生成房间的名称
    例如，scene_type = kitchen的第num = 5个房间，为FloorPlan5_physics
    """
    mapping = {"kitchen":'', "living_room":'2', "bedroom":'3', "bathroom":'4'}
    front = mapping[scene_type]
    endd = '_physics' if (front == '' or front == '2') else ''
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num) + endd
    return "FloorPlan" + front + "0" + str(num) + endd

def random_divide(total_epi, chosen_scenes, n, shuffle = True):
    """
    total_epi是个整数，是一共需要训练的epi数，chosen_scenes需要是一个dict，
    键值为任意字符串组成的list
    函数会把chosen_scenes伪平分为n份，同时把total_epi平分为对应的n份
    shuffle决定了是否要随机重排
    """
    scenes = [x for i in chosen_scenes.values() for x in i]
    out = []
    if shuffle: random.shuffle(scenes)
    if n > len(scenes):
        epi_nums = [total_epi//n for _ in range(n)]
        for i in range(0, total_epi%n):
            epi_nums[i%n]+=1
        out = [scenes for _ in range(n)]
        return out, epi_nums
    step = len(scenes)//n
    mod = len(scenes)%n
    
    for i in range(0, n*step, step):
        out.append(scenes[i:i + step])
    
    for i in range(0, mod):
        out[i].append(scenes[-(i+1)])

    num_per_epi = total_epi/len(scenes)
    epi_nums = [round(len(x)*num_per_epi) for x in out]
    epi_nums[0] += total_epi-sum(epi_nums)
    if not shuffle: epi_nums = [total_epi//n for _ in range(n)]
    return out, epi_nums

def get_test_set(args):
    """
    返回scene_names_div,每个线程需要测试的房间名
    chosen_objects,不同类别场景需要测试的目标，这个变量对每个线程都一样的
    nums_div,每个线程要测试的epi数量
    test_set_div,当导入了测试序列文件时，为每个线程的测试序列
    """
    sche = {}
    shuffle = args.shuffle
    chosen_scene_names = get_scene_names(args.test_scenes)
    chosen_objects = args.test_targets
    test_set_div = None
    if args.test_sche_dir == '':
        scene_names_div, nums_div = random_divide(
            args.total_eval_epi, chosen_scene_names, args.threads, shuffle
            )
    else:
        print('Using Test Schedule at ',args.test_sche_dir)
        total_epi = 0
        scene_names_div = []
        #是按照chosen scene names指定的房间类型加载json的，所以不能只单单指定一个路径
        for k in chosen_scene_names:
            pa = os.path.join(args.test_sche_dir,k+'_test_set.json')
            import json
            with open(pa, 'r') as f:
                sche[k] = json.load(f)
            total_epi += len(sche[k])
        args.total_eval_epi = total_epi
        test_set_div , nums_div = random_divide(total_epi, sche, args.threads, shuffle)
        for i in range(args.threads):
            scene_names_div.append(set())
            for x in test_set_div[i]:
                scene_names_div[i].add(x[0])
            scene_names_div[i] = list(scene_names_div[i])
    return scene_names_div, chosen_objects, nums_div, test_set_div

if __name__ == "__main__":
    train_scenes = {
        'kitchen':'21-30',
        'living_room':'21-30',
        'bedroom':'21-30',
        'bathroom':'21-30',
    }
    cc = get_scene_names(train_scenes)
    name_div, num_div = random_divide(1000, cc, 4, False)
    print(name_div, num_div)

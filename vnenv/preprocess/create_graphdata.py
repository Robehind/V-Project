import os
from tqdm import tqdm
import random
import networkx.readwrite as netx
import json
"""生成我的脚本要用的trans.json文件"""
"""trans.json保存了每一个状态是否可以向前走一步"""
def write_to_json(ips, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ips, f)

def get_scene_names(train_scenes):
    
    return [
        make_scene_name(k, i) for k in train_scenes.keys() for i in train_scenes[k]
    ]

def make_scene_name(scene_type, num):
    mapping = {"kitchen":'', "living_room":'2', "bedroom":'3', "bathroom":'4'}
    front = mapping[scene_type]
    endd = '_physics' if (front == '' or front == '2') else ''
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num) + endd
    return "FloorPlan" + front + "0" + str(num) + endd

def get_coor_str(state):
    a,b,_,_ = state.split('|')
    return a+'|'+b

move_list = [0, 1, 1, 1, 0, -1, -1, -1]
move_list = [x*0.25 for x in move_list]

def make_graph(scene_name):
    #print("making ",scene_name)
    datadir = '../mixed_offline_data/'
    
    
    with open(os.path.join(datadir, scene_name, 'graph.json'),"r",) as f:
        graph_json = json.load(f)
    graph = netx.node_link_graph(graph_json).to_directed()
    all_states = list(graph.nodes())
    
    data = {x:0 for x in all_states}
    save_str = os.path.join(datadir, scene_name, 'trans.json')

    for k in all_states:
        #coor_str = get_coor_str(k)
        x, z, rot, hor = k.split('|')
        x,z,rot,hor = float(x),float(z),int(rot),int(hor)
        
        x += move_list[(rot//45)%8]
        z += move_list[((rot//45)+2)%8]
        
        neighbors = graph.neighbors(k)
        #neighbor_coor = [get_coor_str(n) for n in neighbors]
        if "{:0.2f}|{:0.2f}|{:d}|{:d}".format(x,z,rot,hor) in neighbors:
            data[k] = 1

    write_to_json(data, save_str)


test_scenes = {
        'kitchen':range(1,31),
        'living_room':range(1,31),
        'bedroom':range(1,31),
        'bathroom':range(1,31),
    }

scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')

p = tqdm(total = len(scene_names))
#258154
for n in scene_names:
    p.update(1)
    make_graph(n)
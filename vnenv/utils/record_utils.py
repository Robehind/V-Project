import os
from typing import Union, List, Dict
import json


class MeanCalcer(object):
    """Count and sum records in Dict. Average when pop"""
    # TODO list和数字可以都处理成list。
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add(
        self,
        records: Dict[str, Union[List, int, float]],
        count: bool = True
    ):
        for k in records:
            if k not in self._sums:
                self._sums[k] = records[k]
                if isinstance(records[k], list):
                    self._counts[k] = [1]*len(records[k])
                else:
                    self._counts[k] = 1
            else:
                if isinstance(records[k], list):
                    self._list_add(k, records[k])
                else:
                    self._sums[k] += records[k]
                    self._counts[k] += count

    def _list_add(self, label, in_list):
        a = self._sums[label]
        b = in_list
        c = self._counts[label]
        # list长度可以继续变长？为了计算false action ratio
        if len(a) < len(b):
            c += [1 for _ in range(len(a), len(b))]
            a, b = in_list, self._sums[label]
        for i in range(len(b)):
            c[i] += 1
        self._counts[label] = c
        tmp = [a[i]+b[i] for i in range(len(b))] + a[len(b):len(a)]
        self._sums[label] = tmp

    def report(self, no_div_list=[]):
        for k in no_div_list:
            if not isinstance(self._counts[k], list):
                self._counts[k] = 1
            else:
                self._counts[k] = [1 for _ in range(len(self._counts[k]))]
        means = {}
        for k in self._sums:
            if isinstance(self._sums[k], list):
                means[k] = [
                    a/b for a, b in zip(self._sums[k], self._counts[k])
                ]
            else:
                means[k] = self._sums[k] / self._counts[k]
        return means

    def reset(self):
        self._sums = {}
        self._counts = {}

    def pop(self, no_div_list=[]):
        means = self.report(no_div_list)
        self.reset()
        return means


class LabelMeanCalcer(object):
    """带标签的用来算平均数的一个类，记录数据用"""
    def __init__(self):
        self.trackers = {}

    def __getitem__(self, key):
        if key in self.trackers:
            return self.trackers[key]
        else:
            self.trackers[key] = MeanCalcer()
            return self.trackers[key]

    def items(self):
        return self.trackers.items()

    def pop(self, no_div_list=[]):
        out = {}
        for k in self.trackers:
            out[k] = self.trackers[k].pop(no_div_list)
        self.trackers = {}
        return out


def thor_data_output(args, test_scalars):
    """整理数据并输出到json。输入的这个是一个dict，键有房间名称，
    以及前缀一个场景类型的目标字符串(为了告诉函数这个目标是在哪个房间找的)"""
    def get_type(scene_name):
        """根据房间名称返回该房间属于哪个类型"""
        mapping = {'2': 'living_room', '3': 'bedroom', '4': 'bathroom'}
        num = scene_name.split('_')[0].split('n')[-1]
        if len(num) < 3:
            return 'kitchen'
        return mapping[num[0]]
    total_scalars = LabelMeanCalcer()
    scene_split = {k: {} for k in args.test_scenes}
    target_split = {k: {} for k in args.test_scenes}
    result = test_scalars.pop_and_reset(['epis'])

    for k in result:
        k_sp = k.split('/')
        if len(k_sp) == 1:
            s_type = get_type(k_sp[0])
            scene_split[s_type][k] = result[k].copy()
            total_scalars[s_type].add_scalars(result[k])
            total_scalars['Total'].add_scalars(result[k])
        else:
            target_split[k_sp[0]][k_sp[-1]] = result[k].copy()
            total_scalars[k_sp[-1]].add_scalars(result[k])

    total_scalars = total_scalars.pop_and_reset(['epis'])

    for k in scene_split:
        out = dict(Total=total_scalars.pop(k))
        for i in sorted(scene_split[k]):
            scene_split[k][i].pop('false_action_ratio')
            out[i] = scene_split[k][i]
        for i in sorted(target_split[k]):
            target_split[k][i].pop('false_action_ratio')
            out[i] = target_split[k][i]
        result_path = os.path.join(args.exp_dir, k+'_'+args.results_json)
        with open(result_path, "w") as fp:
            json.dump(out, fp, indent=4)

    out = dict(Total=total_scalars.pop('Total'))
    for i in sorted(total_scalars):
        total_scalars[i].pop('false_action_ratio')
        out[i] = total_scalars[i]
    result_path = os.path.join(args.exp_dir, 'Total_'+args.results_json)
    with open(result_path, "w") as fp:
        json.dump(out, fp, indent=4)


def data_output(path: str, filename: str, test_scalars: MeanCalcer):
    result = test_scalars.pop(['epis'])
    result_path = os.path.join(path, 'Total_'+filename)
    with open(result_path, "w") as fp:
        json.dump(result, fp, indent=4)

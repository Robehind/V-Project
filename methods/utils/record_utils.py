from typing import Union, List, Dict
from tensorboardX import SummaryWriter


class MeanCalcer(object):
    """Count and sum records in Dict. Average when pop"""
    # TODO list和数字可以都处理成list。
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def keys(self) -> list:
        return list(self._sums.keys())

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

    def __getitem__(self, key) -> MeanCalcer:
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
            if k in no_div_list:
                out[k] = self.trackers[k].pop(self.trackers[k].keys())
            else:
                out[k] = self.trackers[k].pop()
        self.trackers = {}
        return out


def add_eval_data_seq(
    writer: SummaryWriter,
    data: Dict,
    step: int,
    prefix: str = ''
):
    for k, v in data.items():
        if k == 'trajs':  # TODO trajs pop out to store in json
            continue
        if isinstance(v, dict):
            if prefix != '':
                raise Exception("data dict too deep(Can't Dict[Dict[Dict]])")
            else:
                add_eval_data_seq(writer, v, step, k)
        elif isinstance(v, list):
            tmp = k if prefix == '' else prefix+'/'+k
            if type(v[0]) not in [list, tuple]:
                v = list(enumerate(v, 1))
            for x, y in v:
                writer.add_scalars(tmp, {str(step): y}, x)
        else:
            if prefix == '':
                writer.add_scalar(k, v, step)
            else:
                writer.add_scalars(prefix, {k: v}, step)


def add_eval_data(writer: SummaryWriter, data: Dict, prefix: str = ''):
    """add data organized in Dict to TensoboardX"""
    result_str = ''
    for k, v in data.items():
        if k == 'trajs':  # TODO trajs pop out to store in json
            continue
        if isinstance(v, dict):
            if prefix != '':
                raise Exception("data dict too deep(Can't Dict[Dict[Dict]])")
            else:
                add_eval_data(writer, v, k)
        elif isinstance(v, list):
            if type(v[0]) not in [list, tuple]:
                v = list(enumerate(v, 1))
            for x, y in v:
                if prefix == '':
                    writer.add_scalar(k, y, x)
                else:
                    writer.add_scalars(prefix, {k: y}, x)
        else:
            result_str += f'{k}: {v}<br>'
    tag = 'result' if prefix == '' else prefix
    if result_str != '':
        writer.add_text(tag, result_str[:-4])

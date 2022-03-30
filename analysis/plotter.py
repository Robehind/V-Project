from matplotlib import pyplot as plt
from methods.utils.record_utils import LabelMeanCalcer, MeanCalcer
from epidata import EpisodeData
from metrics import metrics1, metrics2
from matplotlib.widgets import TextBox, RadioButtons, Button
import argparse
import seaborn
from taskenvs.ai2thor_env import OriThorForVis
import numpy as np
import os
from mplcursors import cursor
x_label = ['scene', 'target', 'model', 'min_acts', 'steps']
all_metrics = metrics1 + metrics2
GRIDSIZE = 0.25  # TODO


def init_parser():
    parser = argparse.ArgumentParser(description="Plot curves and heatmap")
    parser.add_argument(
        "--path", type=str, default='/home/zhiyu/',
        help="Path to the evaluated dir")
    parser.add_argument(
        "--width", type=int, default=640,
        help="Width of top view image. Height will be the same.")
    args = parser.parse_args()
    return args


def _pose2picXY(cam_params, height, width):
    # TODO 似乎高和宽必须得是一样的.
    cam_pos = cam_params['position']
    orth_size = cam_params['orthographicSize']
    lower_left = np.array((cam_pos['x'], cam_pos['z'])) - orth_size

    def pose2picXY(x, z):
        nx, nz = (np.array((x, z)) - lower_left) / (2*orth_size)
        return round(height*(1. - nz)), round(width*nx)
    return pose2picXY


class Plotter:
    def __init__(
        self,
        trajs_path: str
    ) -> None:
        # 载入轨迹数据
        self.data = EpisodeData(trajs_path)
        # 参数
        self.regular_dict = {}
        self.X_axis, self.last_X_axis = ['scene']*2
        self.Y_axis, self.last_Y_axis = ['SR']*2
        # 生成画布和相关组建
        fig, self.ax = plt.subplots()
        fig.subplots_adjust(left=0.3, bottom=0.3)
        self.ax2 = self.ax.twinx()
        # 指定筛选项目筛选框
        text_axes = [
            fig.add_axes([0.1 + x*0.3, 0.125, 0.2, 0.05])
            for x in [0, 1, 2]]
        txtboxes = {
            s: TextBox(ax, s, textalignment='center')
            for ax, s in zip(text_axes, ['scene', 'target', 'model'])}
        [v.on_submit(self.creat_submit(k)) for k, v in txtboxes.items()]
        [v.set_val('.') for v in txtboxes.values()]
        # 自由字典正则筛选框
        ax_dict = fig.add_axes([0.1, 0.05, 0.8, 0.05])
        txt_other = TextBox(ax_dict, 'Other', textalignment='center')
        txt_other.on_submit(self.other_submit)
        txt_other.set_val("")
        # 横坐标选择框
        ax_x = fig.add_axes([0.05, 0.73, 0.08, 0.15])
        x_choose = RadioButtons(ax_x, x_label)
        x_choose.on_clicked(self.choose_xaxis)
        x_choose.set_active(0)
        self.x_choose = x_choose
        # 纵坐标选择框
        ax_y = fig.add_axes([0.14, 0.64, 0.1, 0.24])
        y_choose = RadioButtons(ax_y, all_metrics)
        y_choose.on_clicked(self.choose_yaxis)
        y_choose.set_active(0)
        self.y_choose = y_choose
        # 表格数据展示
        self.ax_table = fig.add_axes([0.05, 0.39, 0.19, 0.23])
        self.draw_table()
        # 开始绘画曲线按钮
        ax_start = fig.add_axes([0.05, 0.3, 0.09, 0.07])
        draw_button = Button(ax_start, 'Draw\nCurve')
        draw_button.on_clicked(self.plot)
        # 开始绘画热力图按钮
        ax_start2 = fig.add_axes([0.15, 0.3, 0.09, 0.07])
        draw_button2 = Button(ax_start2, 'Draw\nHeatmap')
        draw_button2.on_clicked(self.plot_heat)
        # show
        plt.show()

    def creat_submit(self, name):
        def func(scene):
            self.regular_dict[name] = scene
        return func

    def choose_xaxis(self, label):
        self.X_axis = label

    def choose_yaxis(self, label):
        self.Y_axis = label

    def other_submit(self, dd):
        if dd == '':
            to_pop = []
            for k in self.regular_dict:
                if k in ['model', 'scene', 'target']:
                    continue
                to_pop.append(k)
            [self.regular_dict.pop(k) for k in to_pop]
        else:
            for ss in dd.split(" "):
                k, v = ss.split(":")
                if k in ['model', 'scene', 'target']:
                    # 优先级比专用筛选更低
                    continue
                self.regular_dict[k] = v

    def resume_xy(self):
        self.x_choose.set_active(x_label.index(self.last_X_axis))
        self.y_choose.set_active(all_metrics.index(self.last_Y_axis))

    def plot_heat(self, event):
        if not hasattr(self, 'ctrler'):
            self.venv = OriThorForVis(
                width=args.width, height=args.width,
                grid_size=GRIDSIZE, rotate_angle=45)
        hgz = GRIDSIZE / 2.
        epis = self.data.get_episodes(self.regular_dict)
        scene = epis[0]['scene']
        cam_params, pic = self.venv.top_view(scene)
        tfunc = _pose2picXY(cam_params, pic.shape[0], pic.shape[1])
        fig, ax = plt.subplots()
        heatmat = np.zeros((pic.shape[0], pic.shape[1]))
        for epi in epis:
            assert scene == epi['scene']
            for p in epi['poses']:  # TODO Done动作会导致多一个重复的pose
                x, z, _, _ = [float(x) for x in p.split("|")]
                # TODO
                ULx, ULz = tfunc(x - hgz, z + hgz)
                BRx, BRz = tfunc(x + hgz, z - hgz)
                heatmat[ULx:BRx, ULz:BRz] += 1
        ax.imshow(pic[:, :, ::-1])
        ax.axis('off')
        seaborn.heatmap(
            heatmat,
            mask=heatmat == 0,
            cmap=plt.get_cmap('OrRd'),
            alpha=0.5, robust=True)
        plt.show()

    def draw_table(self, avedata=[0]*len(metrics1)):
        half = len(metrics1) // 2
        table_value = [
            metrics1[:half], avedata[:half],
            metrics1[half:], avedata[half:]]
        if len(metrics1) % 2 != 0:
            table_value[0].append("")
            table_value[1].append("")
        table = self.ax_table.table(
            cellText=table_value,
            bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        self.ax_table.axis('off')

    def plot(self, event):
        if hasattr(self, 'cursor'):
            self.cursor.remove()
        # 检查坐标选择合法性
        if (self.X_axis == 'steps' and self.Y_axis in metrics1) or \
           (self.X_axis != 'steps' and self.Y_axis in metrics2):
            print(f"Can't choose {self.Y_axis} when x-axis is {self.X_axis}")
            self.resume_xy()
            return
        epis = self.data.get_episodes(self.regular_dict)
        # 重画表格
        self.ax_table.cla()
        ave_data = [0.0]*len(metrics1)
        for ep in epis:
            for i, x in enumerate(metrics1):
                ave_data[i] += ep[x]
        self.draw_table([
            round(x/len(epis), 4) for x in ave_data])
        # 会先全部清除然后重画
        self.ax.cla()
        self.ax2.cla()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_visible(False)
        if self.X_axis == 'steps':
            self.ax2.set_visible(True)
            line = self.DrawStepsCurve(epis)
        elif self.X_axis == 'min_acts':
            line = self.DarwMinActsCurve(epis)
        else:
            line = self.DrawNormalCurve(epis)
        # 显示数据的鼠标
        self.cursor = cursor(line, hover=True)
        self.ax.set_xlabel(self.X_axis)
        self.ax.set_ylabel(self.Y_axis)
        self.last_X_axis = self.x_choose.value_selected
        self.last_Y_axis = self.y_choose.value_selected
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()

    def DrawStepsCurve(self, epis):
        # 横坐标为steps的曲线数据，同时返回一个样本数量分布
        y_axis = self.Y_axis
        tracker = MeanCalcer()
        assert y_axis in metrics2
        for e in epis:
            data = e[y_axis]
            tracker.add({y_axis: data})
        # 样本数量
        sample_nums = tracker._counts[y_axis].copy()
        Y = tracker.pop()[y_axis]
        self.ax2.bar(range(1, len(Y)+1), sample_nums, color='gray', alpha=.3)
        self.ax2.set_ylabel('sample nums')
        line, = self.ax.plot(range(1, len(Y)+1), Y, marker='o', markersize=3)
        return line

    def DarwMinActsCurve(self, epis):
        x_axis, y_axis = self.X_axis, self.Y_axis
        assert x_axis in x_label
        tracker = MeanCalcer()
        assert y_axis in metrics1
        for e in epis:
            label = e[x_axis]
            data = [e[y_axis]]*int(label)
            tracker.add({y_axis: data})
        Y = tracker.pop()[y_axis]
        line, = self.ax.plot(range(1, len(Y)+1), Y, marker='o', markersize=3)
        self.ax.set_xticks(range(1, len(Y)+1), rotation=0)
        return line

    def DrawNormalCurve(self, epis):
        x_axis, y_axis = self.X_axis, self.Y_axis
        assert x_axis in x_label
        tracker = LabelMeanCalcer()
        assert y_axis in metrics1
        for e in epis:
            label = e[x_axis]
            data = e[y_axis]
            tracker[str(label)].add({y_axis: data})
        data = tracker.pop()
        if x_axis == 'scene':
            X = sorted(
                data.keys(),
                key=lambda x: int(x.split("_")[0].split('n')[-1]))
            rot = 30
        elif x_axis == 'model':
            X = sorted(data.keys(), key=lambda x: int(x.split('_')[-1]))
            rot = 30
        else:
            X = sorted(data.keys())
            rot = 30
        Y = [data[x][y_axis] for x in X]
        line, = self.ax.plot(range(len(X)), Y, marker='o', markersize=3)
        # TODO
        if x_axis == 'scene':
            X = [x.split("_")[0] for x in X]
        self.ax.set_xticks(range(len(X)), X, rotation=rot)
        return line


if __name__ == '__main__':
    args = init_parser()
    trajs_path = os.path.join(args.path, 'trajs.json')
    plter = Plotter(trajs_path)

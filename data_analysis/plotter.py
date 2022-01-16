from matplotlib import pyplot as plt
from vnenv.utils.record_utils import LabelMeanCalcer, MeanCalcer
from epidata import EpisodeData
from metrics import metrics1, metrics2
from matplotlib.widgets import TextBox, RadioButtons, Button
x_label = ['scene', 'target', 'model', 'steps', 'min_acts']
all_metrics = metrics1 + metrics2


class Plotter:
    def __init__(
        self,
        trajs_path: str
    ) -> None:
        # 载入轨迹数据
        self.data = EpisodeData(trajs_path)
        # 参数
        self.regular_dict = {}
        self.X_axis = 'scene'
        self.Y_axis = 'SR'
        self.last_X_axis = 'scene'
        self.last_Y_axis = 'SR'
        # 生成画布和相关组建
        fig, self.ax = plt.subplots()
        self.fig = fig
        fig.subplots_adjust(left=0.3, bottom=0.3)
        self.ax2 = self.ax.twinx()
        # 指定筛选项目筛选框
        text_axes = [
            fig.add_axes([0.1 + x*0.3, 0.135, 0.2, 0.075])
            for x in [0, 1, 2]
        ]
        txtboxes = {
            s: TextBox(ax, s, textalignment='center')
            for ax, s in zip(text_axes, ['scene', 'target', 'model'])
        }
        [v.on_submit(self.creat_submit(k)) for k, v in txtboxes.items()]
        [v.set_val('.') for v in txtboxes.values()]
        self.txtboxes = txtboxes
        # 自由字典正则筛选框
        ax_dict = fig.add_axes([0.1, 0.05, 0.8, 0.075])
        txt_other = TextBox(ax_dict, 'Other', textalignment='center')
        txt_other.on_submit(self.other_submit)
        txt_other.set_val("")
        self.txt_other = txt_other
        # 横坐标选择框
        ax_x = fig.add_axes([0.05, 0.75, 0.15, 0.15])
        x_choose = RadioButtons(ax_x, x_label)
        x_choose.on_clicked(self.choose_xaxis)
        x_choose.set_active(0)
        self.x_choose = x_choose
        # 纵坐标选择框
        ax_y = fig.add_axes([0.05, 0.47, 0.15, 0.25])
        y_choose = RadioButtons(ax_y, all_metrics)
        y_choose.on_clicked(self.choose_yaxis)
        y_choose.set_active(0)
        self.y_choose = y_choose
        # 开始绘图按钮
        ax_start = fig.add_axes([0.05, 0.3, 0.15, 0.15])
        draw_button = Button(ax_start, 'Draw')
        draw_button.on_clicked(self.plot)
        self.draw_button = draw_button
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

    def plot(self, event):
        # 检查坐标选择合法性
        if (self.X_axis == 'steps' and self.Y_axis in metrics1) or \
           (self.X_axis != 'steps' and self.Y_axis in metrics2):
            print(f"Can't choose {self.Y_axis} when x-axis is {self.X_axis}")
            self.resume_xy()
            return
        epis = self.data.get_episodes(self.regular_dict)
        # 会先全部清除然后重画
        self.ax.cla()
        self.ax2.cla()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.set_visible(False)
        if self.X_axis == 'steps':
            self.ax2.set_visible(True)
            self.DrawStepsCurve(epis)
        elif self.X_axis == 'min_acts':
            self.DarwMinActsCurve(epis)
        else:
            self.DrawNormalCurve(epis)
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
        self.ax.plot(range(1, len(Y)+1), Y)
        self.ax2.bar(range(1, len(Y)+1), sample_nums)
        self.ax2.set_ylabel('sample nums')

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
        self.ax.plot(range(1, len(Y)+1), Y)
        self.ax.set_xticks(range(1, len(Y)+1), rotation=0)

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
        self.ax.plot(range(len(X)), Y)
        # TODO
        if x_axis == 'scene':
            X = [x.split("_")[0] for x in X]
        self.ax.set_xticks(range(len(X)), X, rotation=rot)


if __name__ == '__main__':
    import sys
    json_path = sys.argv[1]
    plter = Plotter(json_path)

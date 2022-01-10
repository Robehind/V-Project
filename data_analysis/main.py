import matplotlib.pyplot as plt
import sys
from epidata import EpisodeData
from plotter import Plotter
from matplotlib.widgets import TextBox, RadioButtons
from metrics import metrics2, metrics1


regular_dict = {}
X_axis = 'scene'
Y_axis = 'SR'
last_X_axi


def choose_xaxis(label):
    global X_axis
    X_axis = label
    epis = data.get_episodes(regular_dict)
    pl.plot(epis, X_axis, Y_axis)


def choose_yaxis(label):
    global Y_axis, X_axis
    if X_axis == 'steps':
        if label in metrics1:
            print("Can't choose that Metric when x-axies is steps")
            y_choose.set_active
            return
    Y_axis = label
    epis = data.get_episodes(regular_dict)
    pl.plot(epis, X_axis, Y_axis)


def creat_submit(name):
    def func(scene):
        regular_dict[name] = scene
        epis = data.get_episodes(regular_dict)
        pl.plot(epis, X_axis, Y_axis)
    return func


def other_submit(dd):
    if dd == '':
        to_pop = []
        for k in regular_dict:
            if k in ['model', 'scene', 'target']:
                continue
            to_pop.append(k)
        [regular_dict.pop(k) for k in to_pop]
    else:
        for ss in dd.split(" "):
            k, v = ss.split(":")
            if k in ['model', 'scene', 'target']:
                # 优先级比专用筛选更低
                continue
            regular_dict[k] = v
    epis = data.get_episodes(regular_dict)
    pl.plot(epis, X_axis, Y_axis)


# 载入轨迹数据
json_path = sys.argv[1]
data = EpisodeData(json_path)
# 生成画布和相关组建
fig, AX = plt.subplots()
fig.subplots_adjust(left=0.3, bottom=0.3)
pl = Plotter(AX)
# 输入正则表达式的筛选框
text_axes = [
    fig.add_axes([0.1 + x*0.3, 0.135, 0.2, 0.075]) for x in [0, 1, 2]
]
ax_dict = fig.add_axes([0.1, 0.05, 0.8, 0.075])
txtboxes = {
    s: TextBox(ax, s, textalignment='center')
    for ax, s in zip(text_axes, ['scene', 'target', 'model'])
}
txt_other = TextBox(ax_dict, 'Other', textalignment='center')
[v.on_submit(creat_submit(k)) for k, v in txtboxes.items()]
[v.set_val('.') for v in txtboxes.values()]
txt_other.on_submit(other_submit)
txt_other.set_val("")
# 横坐标选择框
ax_x = fig.add_axes([0.05, 0.7, 0.15, 0.15])
x_choose = RadioButtons(
    ax_x, ('scene', 'target', 'model', 'steps', 'min_acts'))
x_choose.on_clicked(choose_xaxis)
x_choose.set_active(0)
# 纵坐标选择框
ax_y = fig.add_axes([0.05, 0.4, 0.15, 0.15])
y_choose = RadioButtons(
    ax_y, metrics1 + metrics2)
y_choose.on_clicked(choose_yaxis)
y_choose.set_active(0)
# 数据筛选

plt.show()

from vnenv.environments.ai2thor_env import OriThorForVis
import json
import copy
import argparse
import os
import time


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--birdView", action='store_true')
    parser.add_argument("--smooth", action='store_true')
    parser.add_argument("--wait", type=float, default=0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    return args


def traj_str(traj):
    return f"{traj['scene']}/{traj['target']}/success:{traj['success']}" + \
        f"/return:{traj['return']}/ep_length:{traj['ep_length']}"


def print_details(traj):
    _tj = copy.deepcopy(traj)
    _tj.pop("actions")
    _tj.pop("events")
    print(json.dumps(_tj, indent=4))


def main(args):
    with open(args.path, "r") as fp:
        trajs = json.load(fp)

    trajs_num = len(trajs)
    print(f"There are {trajs_num} trajectories.")
    pool = trajs.copy()
    # choose an episode to visualize
    page = 0
    q_flag = False
    sj = copy.deepcopy(trajs[0])

    env = OriThorForVis(
        width=args.width, height=args.height,
        grid_size=0.25, rotate_angle=45)
    while 1:
        # filter trajs
        while 1:
            plen = len(pool)
            st, ed = page*10+1, min(page*10+10, plen)
            print(f'Page:{st}-{ed}:')
            for i in range(page*10, ed):
                print(str(i+1)+':'+traj_str(pool[i]))
            ipt = input("Enter a number to choose a traj," +
                        "a key to sort by it('r' to reverse)," +
                        "a (key value) pair to filter," +
                        "'a' or 'd' to roll pages," +
                        "'c' to clear filter," +
                        "'q' to quit\n").split(" ")
            if len(ipt) == 1:
                if ipt[0] == 'd':
                    page = 0 if page*10+10 >= plen else page + 1
                elif ipt[0] == 'a':
                    page -= 1
                    if page == -1:
                        page = plen//10 - (plen % 10 == 0)
                elif ipt[0] == 'r':
                    pool.reverse()
                    page = 0
                elif ipt[0] == 'c':
                    pool = trajs
                    page = 0
                elif ipt[0] == 'q':
                    q_flag = True
                    break
                elif ipt[0].isdigit():
                    idx = int(ipt[0])
                    if idx > ed or idx < st:
                        print(f"Traj {idx} isn't in current page.")
                    else:
                        replay_traj = copy.deepcopy(pool[idx-1])
                        break
                else:
                    if ipt[0] not in sj:
                        print(f"key error: {ipt[0]}")
                    else:
                        pool.sort(key=lambda x: x[ipt[0]])
                        page = 0
            elif len(ipt) == 2:
                k, v = ipt
                if k not in sj:
                    print("key error")
                else:
                    v = [float(v), v] if v.isdigit() else [v]
                    new_pool = [x for x in pool if x[k] in v]
                    sorted(new_pool, key=lambda x: x[k])
                    if new_pool == []:
                        print("nothing left here. Something wrong?")
                    else:
                        pool = new_pool
                        page = 0

        if q_flag:
            return

        print_details(replay_traj)
        key, k2 = 'a', 'a'
        while key not in ['n', 'y']:
            key = input("Replaying this traj:(y/n)")
            if key == 'y':
                env.visualize(
                    replay_traj['scene'], replay_traj['poses'],
                    args.wait, args.birdView, args.smooth)
                while k2 not in ['n', 'y']:
                    k2 = input("Export Video?(y/n)")
                    if k2 == 'y':
                        prefix = input("Input a prefix:")
                        pp, _ = os.path.split(args.path)
                        env.export_video(pp, args.smooth, prefix)
        env.clear_frames()
        time.sleep(1.5)


if __name__ == "__main__":
    args = init_parser()
    main(args)

import os
import cv2
import json
import environments as envs
from environments.env_wrapper import make_envs, VecEnv
from utils.init_func import get_args, get_trajs_path
import copy


def traj_str(traj):
    return f"{traj['scene_id']}/{traj['target']}/success:{traj['success']}" + \
        f"/return:{traj['return']}/ep_length:{traj['ep_length']}"


def main():
    # 读取参数
    args = get_args(os.path.basename(__file__))

    # load env
    env_cls = getattr(envs, args.env)
    args.env_args['obs_args']["obs_dict"] = {"image": "images.hdf5"}
    env_fns = [make_envs(args.env_args, env_cls)]
    Venv = VecEnv(env_fns)
    Venv.update_settings(args.eval_task)

    # load visualize file
    trajs_path = get_trajs_path(args)
    with open(trajs_path, "r") as fp:
        trajs = json.load(fp)

    trajs_num = len(trajs)
    sj = copy.deepcopy(trajs[0])
    sj.pop("actions")
    print(f"There are {trajs_num} trajectories.")
    print("Here is a sample info of a traj(omit the actions):")
    print(json.dumps(sj, indent=4))
    pool = trajs.copy()
    # choose an episode to visualize
    page = 0
    q_flag = False
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
            Venv.close()
            return

        # 按要求初始化环境
        reset_params = dict(scene_id=replay_traj['scene_id'],
                            target_str=replay_traj['target'],
                            agent_state=replay_traj['start_at'])
        actions = replay_traj.pop('actions')
        print("Replaying:")
        print(json.dumps(replay_traj, indent=4))

        obs = Venv.reset(**reset_params)
        for action in actions:
            pic = obs['image'][0][:]
            # RGB to BGR
            pic = pic[:, :, ::-1]
            cv2.imshow("Vis", pic)
            p_key = cv2.waitKey(0)
            if p_key == 27:
                print('Early stop by ESC')
                break
            print(Venv.actions[action])
            obs, _, done, _ = Venv.step([action])
            if done[0]:
                print('Replay finished')
        cv2.waitKey(1500)
        cv2.destroyWindow("Vis")


if __name__ == "__main__":
    main()

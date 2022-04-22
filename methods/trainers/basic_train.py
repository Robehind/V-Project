from typing import Callable
from tqdm import tqdm
import wandb
from utils.record_utils import MeanCalcer
from learners.abs_learner import AbsLearner
from samplers.base_sampler import BaseSampler
from taskenvs import Tasker


def basic_train(
    args,
    sampler: BaseSampler,
    learner: AbsLearner,
    tasker: Tasker,
    val_func: Callable
):
    steps = 0
    update_steps = sampler.batch_size
    total_epis = 0
    print_freq = args.print_freq
    save_freq = args.model_save_freq
    print_gate_steps = print_freq
    save_gate_steps = save_freq
    obj_traker = MeanCalcer()
    projn = args.exps_dir.split('/')[-1]
    name = args.exp_dir.split('/')[-1]
    wandb.init(project=projn, entity='robehind', name=name,
               config=args, dir=args.exp_dir)
    pbar = tqdm(total=args.train_steps, unit='step')

    # if args.val_epi != 0:
    #     val_writer = SummaryWriter(os.path.join(args.exp_dir, 'tblog/val'))
    while steps < args.train_steps:

        batched_exp = sampler.sample()
        obj_salars = learner.learn(batched_exp)
        if tasker.next_tasks(update_steps, sampler.report()):
            sampler.reset()

        pbar.update(update_steps)
        steps += update_steps

        # saving and validating
        if steps >= save_gate_steps:
            save_gate_steps += save_freq
            learner.checkpoint(args.exp_dir, steps)
            # validating
            if args.val_epi != 0:
                # save train task
                o_tasks = sampler.Venv.call('tasks')
                # load validate task
                sampler.Venv.set_attr('tasks', args.val_task)
                sampler.Venv.call('add_extra_info', args.val_extra_info)
                sampler.reset()
                # validate process
                val_data = val_func(
                    sampler.agent, sampler.Venv, args.val_epi)
                # resume train task
                sampler.Venv.set_attr('tasks', o_tasks)
                sampler.Venv.call('add_extra_info', args.train_extra_info)
                sampler.reset()
                learner.model.train()
                # log
                out_data = {}
                for k in val_data:
                    out_data['val/'+k] = val_data[k]
                wandb.log(out_data, step=steps)

        # logging
        obj_traker.add(obj_salars)

        if steps >= print_gate_steps:
            print_gate_steps += print_freq
            record = sampler.pop_records()
            total_epis += record.pop('epis')
            t_data = {'epis': total_epis}
            t_data.update(record)
            t_data.update(obj_traker.pop())
            out_data = {}
            for k in t_data:
                out_data['train/'+k] = t_data[k]
            wandb.log(out_data, step=steps)

    sampler.close()
    pbar.close()

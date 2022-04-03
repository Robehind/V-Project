from typing import Callable
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.record_utils import MeanCalcer, add_eval_data_seq
from learners.abs_learner import AbsLearner
from samplers.base_sampler import BaseSampler
from taskenvs import Tasker


def basic_train(
    args,
    sampler: BaseSampler,
    learner: AbsLearner,
    tasker: Tasker,
    tx_writer: SummaryWriter,
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
    pbar = tqdm(total=args.train_steps, unit='step')

    if args.val_epi != 0:
        val_writer = SummaryWriter(os.path.join(args.exp_dir, 'tblog/val'))
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
                # TODO sampler.Venv.call('add_extra_info', args.calc_spl)
                # validate process
                val_data = val_func(
                    sampler.agent, sampler.Venv, args.val_epi)
                # resume train task
                sampler.Venv.set_attr('tasks', o_tasks)
                sampler.Venv.call('add_extra_info', args.train_extra_info)
                sampler.reset()
                # TODO sampler.Venv.call('add_extra_info', False)
                learner.model.train()
                # log
                add_eval_data_seq(val_writer, val_data, steps)

        # logging
        obj_traker.add(obj_salars)

        if steps >= print_gate_steps:
            print_gate_steps += print_freq
            record = sampler.pop_records()
            total_epis += record.pop('epis')
            tx_writer.add_scalar("n_steps", steps, total_epis)
            for k, v in record.items():
                tx_writer.add_scalar(k, v, steps)
            for k, v in obj_traker.pop().items():
                tx_writer.add_scalar(k, v, steps)

    sampler.close()
    tx_writer.close()
    pbar.close()

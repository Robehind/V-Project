from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.record_utils import MeanCalcer
from learners.abs_learner import AbsLearner
from samplers.base_sampler import BaseSampler
from curriculums.abs_cl import AbsCL


def basic_train(
    args,
    sampler: BaseSampler,
    learner: AbsLearner,
    clscher: AbsCL,
    tx_writer: SummaryWriter
):
    steps = 0
    update_steps = sampler.batch_size
    total_epis = 0
    print_freq = args.print_freq
    save_freq = args.model_save_freq
    print_gate_steps = print_freq
    save_gate_steps = save_freq
    obj_traker = MeanCalcer()
    pbar = tqdm(total=args.total_train_steps)
    while steps < args.total_train_steps:

        batched_exp = sampler.sample()
        obj_salars = learner.learn(batched_exp)
        clscher.next_sche(update_steps, sampler.report())

        pbar.update(update_steps)
        steps += update_steps

        # logging
        for k, v in obj_salars.items():
            obj_traker.add({k: v})

        if steps >= print_gate_steps:
            print_gate_steps += print_freq
            record = sampler.pop_records()
            total_epis += record.pop('epis')
            tx_writer.add_scalar("n_steps", steps, total_epis)
            for k, v in record.items():
                tx_writer.add_scalar(k, v, steps)
            for k, v in obj_traker.pop().items():
                tx_writer.add_scalar(k, v, steps)

        # saving
        if steps >= save_gate_steps:
            save_gate_steps += save_freq
            learner.checkpoint(args.exp_dir, steps)

    sampler.close()
    tx_writer.close()
    pbar.close()

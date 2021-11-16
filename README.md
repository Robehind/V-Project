# V-Project

Temporarily named **vnenv**，short for **V**isual **N**avigation **ENV**ironment.

**IMPORTANT: Don't Push to master branch.**

[toc]

## Installation

- Clone the `V-PROJECT` repository and download [vdata]() in the same directory where you clone this repository.
- Install the required packages by `cd V-PROJECT && pip install -r requirements.txt`.
- Add `V-PROJECT` to your `PYTHONPATH`. Here are some [instructions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux) about how to do it in a conda environment.
- Run `pytest`. If there is no error and test is passed, then the installation is successful.

## Quick Start

V-PROJECT uses **config scripts** to manage and control modules and train/validate/test process of an algorithm. In `demo_args` folder, we have presented some simple demo to illustrate the basics.

- **Preprocess**

The demos use resnet50 features in FloorPlan25. So run the preprocess scripts first to generate the feature file for this scene:

```bash
python vnenv/preprocess/thor_data_ops/resnet_feature.py
```

Take the demo using A2C algorithm and a LSTM model to navigate to Microwave/Sink in FloorPlan25. Open the `demo_args/thor_a2c_demo.py` for more details about it. What are those arguments mean can be found in `utils/default_args.py`.

Once we have a config script, we can use the entry scripts to load these configurations and carry out an train/validate/test process.

- **Training**:

```bash
python vnenv/main_train.py exp_args/thor_a2c_demo.py
```

The training script will create an experiment folder in the path specified by `exps_dir`, `exp_name` and the datetime containing logs, checkpoints, config arguments,  etc. The `tblog` folder in it holds all the logs and can be analysis by `tensorboard`. If you set `val_mode=True` and specify some corresponding arguments, a validating process will also be carried out during training.

- **Testing**:

The evaluation entry script will search for the newest model automatically. If need to test a specific model, modify `load_model_dir` in the config script. Test results logs to the tensorboard files too, and will be stored in a folder whose name starts with `Eval-`.

```bash
python vnenv/main_eval.py exp_args/thor_a2c_demo.py
```

Or you can **test all** models in an experiment directory:

Results will be stored in a folder whose name starts with `EvalAll-`.

```bash
python vnenv/eval_all.py exp_args/thor_a2c_demo.py
```

- **Visualization**:

Before visualization, we need to do a testing with `record_traj=True` in the config script to generate a file named `trajs.json` storing the test trajectories first. Note that the `eval_all` script doesn't generate it whether you turn the `record_traj` on.

```bash
python vnenv/main_vis.py exp_args/thor_a2c_demo.py
```

A simple command line interface allows you to filter and choose a particular trajectory in `trajs.json` to visualize. 

Replace the config script in commands above to run the other two demos.

## Development

- [**Documentation**]()

The documentation introduces modules' functions and presents tutorials about how to implement new instances of a module, which is the key to develop new algorithms in this platform.

![framework](.\pics\framework.jpg)

- **Teamwork development notes**

Need to [learn the usage of Git](https://learngitbranching.js.org/?locale=zh_CN) first(especially about Branch, Pull & Push). Then create a new branch to code. You can push your own branch or merge the master branch anytime. But **DON'T** push to the master branch, any unexpected changes in master will be withdrawed.


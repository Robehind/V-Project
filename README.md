# V-Project

Temporarily named **v-project**.

- [V-Project](#v-project)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Module Usage](#module-usage)
  - [Development](#development)

## Installation

- Clone the `V-PROJECT` repository.

- Install the required packages by `cd V-PROJECT && pip install -r requirements.txt`.

- Add `V-PROJECT` to your `PYTHONPATH`. Here are some [instructions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux) about how to do it in a conda environment.

- Generate the discrete [AI2THOR]() data:

  ```bash
  python taskenvs/ai2thor_env/preprocess/dump_scene.py --frame
  ```

  Data will be generated in `../vdata/thordata`. Read this script to see more data generation options.

- Download word embedding [data]() and place it in `vdata` directory.

- Run `pytest`. If there is no error and test is passed, then the installation is successful.

## Quick Start

V-PROJECT uses **config scripts** to manage and control modules and train/validate/evaluate process of an algorithm. In `demo_args` folder, we have presented some simple demo to illustrate the basics.

- **Preprocess**:

The demos use resnet50 features in FloorPlan25. So run the preprocess scripts first to generate the feature files:

```bash
python taskenvs/ai2thor_env/preprocess/resnet_feature.py --resnet50fc
```

Take the demo using A2C algorithm and a LSTM model to navigate to Microwave/Sink in FloorPlan25. Open the `demo_args/thor_a2c_demo.py` for more details. What those arguments mean can be found in `utils/default_args.py`.

Once we have a config script, we can use the entry scripts to load these configurations and carry out a train/validate/evaluate process.

- **Training**:

```bash
python methods/main_train.py demo_args/thor_a2c_demo.py
```

The training script will create an experiment folder in the path specified by `exps_dir`, `exp_name` and the datetime containing logs, checkpoints, config arguments,  etc. The `tblog` folder in `exps_dir` holds all the logs and can be analysis by `tensorboard`. If you set `val_mode=True` and specify some corresponding arguments, a validating process will also be carried out during training.

- **Testing**:

The evaluation entry script will search for the newest model automatically. If need to test a specific model, modify `load_model_dir` in the config script. Test results logs to the tensorboard files too, and will be stored in a folder whose name starts with `Eval-`.

```bash
python methods/main_eval.py demo_args/thor_a2c_demo.py
```

Or you can **test all** models in an experiment directory:

Results will be stored in a folder whose name starts with `EvalAll-`.

```bash
python methods/eval_all.py demo_args/thor_a2c_demo.py
```

- **Result Analysis**

V-project provides a analysis script based on [Matplotlib]() and [seaborn]() to draw curves about some metrics and heatmap (only support AI2THOR env for now)
```bash
python data_analysis/plotter.py --path=<PATH-TO-THE-EVAL-DIR>
```
Click what you want to click :)
- **Visualization**:

Run:

```bash
python data_analysis/visualize.py --path=<PATH-TO-THE-EVAL-DIR> --smooth --birdView
```
A simple command line interface allows you to filter and choose a particular trajectory to visualize. 
Only support AI2THOR for now. Use `--smooth` option to display the smooth movement of a single action (takes more time to render extra frames). Use `--birdView` option to display a top-down view of the scene.
After the visualization, you can choose to save the video of it in the `<PATH-TO-THE-EVAL-DIR>`.

## Module Usage

You can also use modules of V-PROJECT to construct your own train/validate/evaluate process. Here is an example for cart pole in `exp4usage.py`.
```bash
python exp4usage.py
```

## Development

- [**Documentation**]()

The documentation introduces modules' functions and presents tutorials about how to implement new instances of a module, which is the key to develop new algorithms in this project.


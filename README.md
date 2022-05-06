# wasp-AI-ML-m1
Vacuum Cleaning Agent Programming Exercise. 
Training an agent with [REINFORCE](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) Reinforcement Learning algorithm to perform the cleaning task of multi-variate environments. 
## Installation

```
conda env create --name <env_name> --file requirements.yml
```

## Current status

* [x] Simulator
  * [x] agent in the environment
  * [x] obstacles in the environment
  * [x] dirty in the environment 
* [x] Training smart agent 
* [x] Testing smart agent according to users' inputs

## Traininig Instructions

**Constraints/assumptions**:

* there is only one robot
* there is a fixed amount of obstacles =  `num_cells`
* the range of dirty cells range from 1 to the covered cells from the robot location

First, we need to train the agent to become smart. Therefore, we will train the agent in a vast number of different scenarios, where the rewards stand for:

* hitting an obstacle = -0.1
* cleaning dirty cells = +2
* moving one cell = -0.1

This means that every time the agent moves gets a penalty of 0.1. Whenever it hits an obstacle, it gets a penalty of the same amount. Finally, per each cleaned cell of the environment, it gets a reward of +2.  

To do so, there is a [cfg file](https://github.com/tmralmeida/wasp-AI-ML-m1/blob/main/cfg/vpg_training.yaml) that the user can modify 
to set different hyperparameters s.t:


| Parameter Name     |   Type   |    Default    | Additional Info                         |
| ------------------ | :------: | :-----------: | --------------------------------------- |
| epochs             |  `int`   |    1200     | Number of epochs to train                 |
| steps_per_epoch    |  `int`   |    4000     | Number maximum of (s, a) per epoch        |
| max_ep_len         |  `int`   |    1000     | Max len of a traj/episode/rollout         |
| v_train_iters      |  `int`   |     80`     | Number of updates in the value function   |
| obs_reward         | `float`  |     -0.1`   | Reward for hitting an obstacle            |
| dirt_reward        | `float`  |     2.0`    | Reward for cleaning a dirty cell          |
| ene_reward         | `float`  |     -0.1    | Reward for moving one cell                |
| gamma              | `float`  |    0.99     | Discount factor (adv. function)           |
| lambda             | `float`  |    0.97     | Adv. function hyperparameter              |
| pi_lr              | `float`  |    3e-4     | Learning rate for the policy net          |
| v_lr               | `float`  |    1e-3     | Learning rate for the value function      |
| hidden_sizes       | `List`   |   [64,32]   | Shape of each hidden FC layer             |
| window_size        | `int`    |    3        | Robot's perception window                 |


For training, run [train.py](https://github.com/tmralmeida/wasp-AI-ML-m1/blob/main/train.py) script as follows:

```
python train.py --num_cells NUM_CELLS --cfg PATH_TO_CFG
```

The number of side cells can vary within the range `[5,11]`.
After training, check plots and trained models in outputs directory.

## Testing Instructions

For testing, first, change the output model path (`save_dir`) in the [cfg file](https://github.com/tmralmeida/wasp-AI-ML-m1/blob/main/cfg/vpg_training.yaml). Then, run [run_agent.py](https://github.com/tmralmeida/wasp-AI-ML-m1/blob/main/run_agent.py) script as follows:


```
python run_agent.py --num_cells NUM_CELLS --cfg PATH_TO_CFG
```
After launching the simulator, the user should use the buttons of his mouse:
* The user should use the button left  of his mouse to place/remove the robot in/from the environment (black cells)
* The user should use the button right of his mouse to place/remove obstacles (red cells) and dirty in/from the environment (green cells)
* Nevertheless, the user can remove any object by simply clicking again on the respective occupied cell with the correspondent button
* After designing the environment, click `p` to run the training of the smart agent.


The user can keep track of the changes of the environment by looking at the logs in the terminal. 


<video src='docs/7_7_simple.mp4' width=180/>


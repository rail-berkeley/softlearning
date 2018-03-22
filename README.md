# Soft Q-Learning
Soft Q-learning (SQL) is a deep reinforcement learning framework for training maximum entropy policies in continuous domains. The algorithm is based on the paper [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165) presented at the International Conference on Machine Learning (ICML), 2017.

# Getting Started

Soft Q-learning can be run either locally or through Docker.

## Prerequisites

You will need to have [Docker](https://docs.docker.com/engine/installation/) and [Docker Compose](https://docs.docker.com/compose/install/) installed unless you want to run the environment locally.

Most of the models require a [MuJoCo](https://www.roboti.us/license.html) license.

## Docker Installation

Currently, rendering of simulations is not supported on Docker due to a missing display setup. As a fix, you can use a [local installation](#local-installation). If you want to run the MuJoCo environments without rendering, the docker environment needs to know where to find your MuJoCo license key (`mjkey.txt`). You can either copy your key into `<PATH_TO_THIS_REPOSITY>/.mujoco/mjkey.txt`, or you can specify the path to the key in your environment variables:

```
export MUJOCO_LICENSE_PATH=<path_to_mujoco>/mjkey.txt
```

Once that's done, you can run the Docker container with

```
docker-compose up
```

Docker compose creates a Docker container named `soft-q-learning` and automatically sets the needed environment variables and volumes.

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it soft-q-learning bash
```

See examples section for examples of how to train and simulate the agents.

To clean up the setup:
```
docker-compose down
```

## Local Installation

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1. Clone rllab
```
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. [Download](https://www.roboti.us/index.html) and copy MuJoCo files to rllab path:
  If you're running on OSX, download https://www.roboti.us/download/mjpro131_osx.zip instead, and copy the `.dylib` files instead of `.so` files.
```
mkdir -p /tmp/mujoco_tmp && cd /tmp/mujoco_tmp
wget -P . https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
mkdir <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libmujoco131.so <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libglfw.so.3 <installation_path_of_your_choice>/rllab/vendor/mujoco
cd ..
rm -rf /tmp/mujoco_tmp
```

3. Copy your MuJoCo license key (mjkey.txt) to rllab path:
```
cp <mujoco_key_folder>/mjkey.txt <installation_path_of_your_choice>/rllab/vendor/mujoco
```

4. Clone `softqlearning`
```
cd <installation_path_of_your_choice>
git clone https://github.com/haarnoja/softqlearning.git
```

5. Create and activate conda environment
```
cd softqlearning
conda env create -f environment.yml
source activate sql
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:
```
source deactivate
conda remove --name sql --all
```

## Examples
### Training and simulating an agent
1. To train the agent
```
python ./examples/mujoco_all_sql.py --env=swimmer --log_dir="/root/sql/data/swimmer-experiment"
```

2. To simulate the agent (*NOTE*: This step currently fails with the Docker installation, due to missing display.)
```
python ./scripts/sim_policy.py /root/sql/data/swimmer-experiment/itr_<iteration>.pkl
```

`mujoco_all_sql.py` contains several different environments and there are more example scripts available in the  `/examples` folder. For more information about the agents and configurations, run the scripts with `--help` flag. For example:
```
python ./examples/mujoco_all_sql.py --help
usage: mujoco_all_sql.py [-h]
                         [--env {ant,walker,swimmer,half-cheetah,humanoid,hopper}]
                         [--exp_name EXP_NAME] [--mode MODE]
                         [--log_dir LOG_DIR]
```
### Training and combining policies
It is also possible to merge two existing maximum entropy policies to form a new composed skill that approximately optimizes both constituent tasks simultaneously as discussed in [ Composable Deep Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/1803.06773). To run the pusher experiment described in the paper, you can first train two policies for the constituent tasks ("push the object to the given x-coordinate" and "push the object to the given y-coordinate") by running 
```
python ./examples/pusher_pretrain.py --log_dir=/root/sql/data/pusher
```
You can then combine the two policies to form a combined skill ("push the object to the given x and y coordinates"), without collecting more experience form the environment, with
```
python ./examples/pusher_combine.py --log_dir=/root/sql/data/pusher/combined \
--snapshot1=/root/sql/data/pusher/00/params.pkl \
--snapshot2=/root/sql/data/pusher/01/params.pkl
```


# Credits
The soft q-learning algorithm was developed by [Haoran Tang](https://math.berkeley.edu/~hrtang/) and [Tuomas Haarnoja](https://people.eecs.berkeley.edu/~haarnoja/) under the supervision of Prof. [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) and Prof. [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) at UC Berkeley. Special thanks to [Vitchyr Pong](https://github.com/vitchyr), who wrote some parts of the code, and [Kristian Hartikainen](https://github.com/hartikainen) who helped testing, documenting, and polishing the code and streamlining the installation process. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).

# References
```
@article{haarnoja2017reinforcement,
  title={Reinforcement Learning with Deep Energy-Based Policies},
  author={Haarnoja, Tuomas and Tang, Haoran and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  year={2017}
}
@article{haarnoja2018composable,
  title={Composable Deep Reinforcement Learning for Robotic Manipulation},
  author={Tuomas Haarnoja, Vitchyr Pong, Aurick Zhou, Murtaza Dalal, Pieter Abbeel, Sergey Levine},
  booktitle={International Conference on Robotics and Automation},
  year={2018}
}

```
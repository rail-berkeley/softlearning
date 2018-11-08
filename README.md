# Soft Reinforcement Learning
Soft Reinforcement Learning is a deep reinforcement learning framework for training maximum entropy policies in continuous domains. The algorithms are based on the following papers:
- [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165) presented at the International Conference on Machine Learning (ICML), 2017.
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://drive.google.com/file/d/0Bxz3x8U2LH_2QllDZVlUQ1BJVEJHeER2YU5mODNaeFZmc3dz/view) presented at the [Deep Reinforcement Learning Symposium](https://sites.google.com/view/deeprl-symposium-nips2017/), NIPS 2017.

This implementation uses Tensorflow. For a PyTorch implementation of soft actor-critic, take a look at [rlkit](https://github.com/vitchyr/rlkit) by [Vitchyr Pong](https://github.com/vitchyr).

# Getting Started

Soft Reinforcement Learning can be run either locally or through Docker.

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

Docker compose creates a Docker container named `soft-TODO-learning` and automatically sets the needed environment variables and volumes.

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it soft-TODO-learning bash
```

See examples section for examples of how to train and simulate the agents.

To clean up the setup:
```
docker-compose down
```

## Local Installation

1. [Download](https://www.roboti.us/index.html)the MuJoCo version 1.50 binaries for Linux or osx. Unzip the downloaded mjpro150 directory into ~/.mujoco/mjpro150.

OR, you can just run the following command:
```
# Mujoco for gym and mujoco_py
export MUJOCO_VERSION=150
export MUJOCO_PATH=~/.mujoco
if [ "$(uname)" == "Darwin" ]; then
    export MUJOCO_TYPE="osx"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    export MUJOCO_TYPE="linux"
fi

MUJOCO_ZIP="mjpro${MUJOCO_VERSION}_${MUJOCO_TYPE}.zip" \
    && mkdir -p ${MUJOCO_PATH} \
    && wget -P ${MUJOCO_PATH} https://www.roboti.us/download/${MUJOCO_ZIP} \
    && unzip ${MUJOCO_PATH}/${MUJOCO_ZIP} -d ${MUJOCO_PATH} \
    && rm ${MUJOCO_PATH}/${MUJOCO_ZIP}
```

2. Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt:

3. Clone `softlearning`
```
cd <installation_path_of_your_choice>
git clone https://github.com/haarnoja/softlearning.git
```

4. Create and activate conda environment
```
cd softlearning
conda env create -f environment.yml
source activate softlearning
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:
```
source deactivate
conda remove --name softlearning --all
```

## Examples
### Training and simulating an agent
1. To train the agent
```
python ./examples/TODO.py --env=swimmer --log_dir="TODO"
```

2. To simulate the agent (*NOTE*: This step currently fails with the Docker installation, due to missing display.)
```
python ./scripts/sim_policy.py TODO/itr_<iteration>.pkl
```

`TODO.py` contains several different environments and there are more example scripts available in the  `/examples` folder. For more information about the agents and configurations, run the scripts with `--help` flag. For example:
```
python ./examples/TODO.py --help
usage: TODO
```
### Training and combining policies
It is also possible to merge two existing maximum entropy policies to form a new composed skill that approximately optimizes both constituent tasks simultaneously as discussed in [ Composable Deep Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/1803.06773). To run the pusher experiment described in the paper, you can first train two policies for the constituent tasks ("push the object to the given x-coordinate" and "push the object to the given y-coordinate") by running
```
python ./examples/pusher_pretrain.py --log_dir=TODO
```
You can then combine the two policies to form a combined skill ("push the object to the given x and y coordinates"), without collecting more experience form the environment, with
```
python ./examples/pusher_combine.py --log_dir=TODO \
--snapshot1=TODO \
--snapshot2=TODO
```


# Credits
- The soft actor-critic algorithm was developed by: [Tuomas Haarnoja](https://people.eecs.berkeley.edu/~haarnoja/), [Aurick Zhou](https://github.com/azhou42), Prof. [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), and Prof. [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)
- The soft q-learning algorithm was developed by: [Haoran Tang](https://math.berkeley.edu/~hrtang/), [Tuomas Haarnoja](https://people.eecs.berkeley.edu/~haarnoja/), Prof. [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), and Prof. [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)

All the soft reinforcement learning algorithms in this package were developed at UC Berkeley. Special thanks to [Vitchyr Pong](https://github.com/vitchyr), who wrote some parts of the code, and [Kristian Hartikainen](https://github.com/hartikainen) who helped testing, documenting, and polishing the code and streamlining the installation process. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).

# References
```
@article{haarnoja2017soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={Deep Reinforcement Learning Symposium},
  year={2017}
}
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

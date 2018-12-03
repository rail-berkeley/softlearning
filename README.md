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

TODO(hartikainen): Fill these in.

### docker-compose
To build the image and run the container:
```
export MJKEY="$(cat ~/.mujoco/mjkey.txt)" && docker-compose -f ./docker/docker-compose.dev.gpu.yml up -d --force-recreate
```

To clean up the setup:
```
docker-compose down
```

### docker run
To build the image and run the container:
- TODO(hartikainen): Fill these in.

To clean up the setup:
```
docker rm -f softlearning
```

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it softlearning bash
```

See examples section for examples of how to train and simulate the agents.

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
git clone https://github.com/rail-berkeley/softlearning.git
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
python ./examples/TODO.py --universe=gym --domain=swimmer --task=default --log_dir="TODO"
```

2. To simulate the agent (*NOTE*: This step currently fails with the Docker installation, due to missing display.)
TODO(hartikainen): update
```
python ./scripts/TODO.py TODO
```

`TODO.py` contains several different environments and there are more example scripts available in the  `/examples` folder. For more information about the agents and configurations, run the scripts with `--help` flag. For example:
```
python ./examples/TODO.py --help
usage: TODO
```

# Credits
TODO(hartikainen): Fill these in.

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

# Soft Q-Learning
Soft Q-Learning is a deep reinforcement learning framework for training expressive, energy-based policies in continuous domains. This implementation is based on [rllab](https://github.com/openai/rllab). Full algorithm is detailed in our paper, [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165), and videos can be found [here](https://sites.google.com/view/softqlearning/home).
# Installation
The implementation is compatible with the rllab interface (see [documentation](https://rllab.readthedocs.io/en/latest/index.html)), and depends on some of its features which are included in this package for convenience. Additionally, some of the examples uses [MuJoCo](http://www.mujoco.org/) physics engine. For installation, you might find [rllab documentation](http://rllab.readthedocs.io/en/latest/user/installation.html) useful. You should add the MuJoCo library files and the key in `/vendor/mujoco` folder.

You will need Tensorflow 1.0 or later. Full list of dependencies is listed in [`requirements.txt`](https://github.com/haarnoja/softqlearning/blob/master/requirements.txt).

# Examples
There are three example environments:
- In the `MultiGoal` environment, task is to move a point-mass into one of four equally good goal locations (see details in [our paper](https://arxiv.org/abs/1702.08165)).
- In the [`Swimmer`](https://gym.openai.com/envs/Swimmer-v1) environment, a two-dimensional, three-link snake needs to learn to swim forwards and backwards.

To train these models run
```
python softqlearning/scripts/learn_<env>.py
```
and to test a trained model, run
```
python softqlearning/scripts/sim_policy.py data/<env>/itr_<#>.pkl
```
where `<env>` is the name of an environment and `<#>` is a iteration number.

# Credits
The Soft Q-Learning package was developed by Haoran Tang and Tuomas Haarnoja, under the supervision of Pieter Abbeel and Sergey Levine, in 2017 at UC Berkeley. We thank Vitchyr Pong and Shane Gu, who helped us implementing some parts of the code. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).

# Reference
```
@article{haarnoja2017reinforcement,
  title={Reinforcement Learning with Deep Energy-Based Policies},
  author={Haarnoja, Tuomas and Tang, Haoran and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  year={2017}
}
```

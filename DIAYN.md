# Diversity Is All You Need
Diversity Is All You Need (DIAYN) is a deep reinforcement learning algorithm for automatically discovering a set of useful skills. The algorithm is based on the paper [Diversity is All You Need: Learning Diverse Skills without a Reward Function](https://arxiv.org/pdf/1802.06070.pdf), with additional videos on the [project website](https://sites.google.com/view/diayn).

# Getting Started

Follow the installation instructions in the [README](./README.md).

### Using Pretrained Skills

Every time DIAYN is run, the set of learned skills is saved in a `.pkl` file. We provide a few utilities for visualizing the learned skills.
The script [visualize_skills.py](./scripts/visualize_skills.py) creates videos of the learned skills.
```
# Usage: python scripts/visualize_skills.py <snapshot.pkl>

python scripts/visualize_skills.py data/demo/half_cheetah/seed_1/itr_3000.pkl
```
Use the `--separate_videos=True` flag to create a separate video for each skill, and use the `--max-path-length=100` to specify the number of steps per episode.

The script [plot_traces.py](./scripts/plot_traces.py) plots the location of the agent throughout an episode.
```
# Usage: python scripts/plot_traces.py <snapshot.pkl>

python scripts/plot_traces.py data/demo/half_cheetah/seed_1/itr_3000.pkl
```
Additional flags allow the user to choose the number of rollouts per skill, the length of each rollout, and which dimensions of the state to plot.

### Training New Skills

Training a new set of skills is as easy as running
```
# Usage: python examples/mujoco_all_diayn.py --env=<ENV> --log_dir=<LOG_DIR>

python examples/mujoco_all_diayn.py --env=half-cheetah --log_dir=data/demo
```
The following environments are currently supported: `swimmer, hopper, walker, half-cheetah, ant, humanoid, point, point-maze, inverted-pendulum, inverted-double-pendulum, mountain-car, lunar-lander, bipedal-walker`. To add a new environment, simply add a new entry to the `ENV_PARAMS` dictionary on line 54 of `mujoco_all_diayn.py`.
The log directory specifies where checkpoints and the `progress.csv` log will be saved. Set this to `/dev/null` if you don't want to save these.
The [rllab](https://github.com/rll/rllab) library has a script [frontend.py](https://github.com/rll/rllab/blob/master/rllab/viskit/frontend.py) that can be used for plotting training in real time (using the `progress.csv` log). Note that the script must be re-run every time the log is updated:
```
# Usage: python rllab/viskit/frontend.py <progress.csv>

python rllab/viskit/frontend.py data/demo/half_cheetah/seed_1/progress.csv
```

### Imitation Experiment

The imitation learning experiments require two checkpoints (`.pkl` files), one to use as the expert and another to use as the student. To run the imitation experiment:
```
# Usage: python scripts/imitate_skills.py --expert_snapshot=<expert.pkl> --student_snapshot=<student.pkl>

python scripts/imitate_skills.py --expert_snapshot=data/demo/half_cheetah/seed_1/itr_2500.pkl --student_snapshot=data/demo/half_cheetah/seed_2/itr_2500.pkl
```

This script saves a JSON file storing the results of the experiment:
* `M` - A 2D array with size (num. expert skills X num. student skills). Entry `M[i, j]` is the predicted probability that student skill `j` matches expert skill `i`.
* `L` - A 2D array with size (num. expert skills X num. student skills). Entry `L[i, j]` is a *list* of the log-probability that student skill `j` matches expert skill `i` at every step in a rollout. Note that rollouts may have different lengths, so do not attempt to reshape `L` into a 3D tensor.
* `dist_vec` - A 1D array with size (num. expert skills). Entry `dist_vec[i]` is the distance between expert skill `i` and the retrieved student skill. Distance is computed as the Euclidean distance between states in a rollout.

### Finetuning Experiment

The finetuning experiments require a checkpoint of saved skills to use as initialization. To run the finetuning experiment:
```
# Usage: python examples/mujoco_all_diayn_finetune.py --env=<ENV> --snapshot=<snapshot.pkl> --log_dir=<log_dir>

python examples/mujoco_all_diayn_finetune.py --env=half-cheetah --snapshot=data/demo/half_cheetah/seed_1/itr_2500.pkl --log_dir=data/finetune/half_cheetah
```

### New Experiments?
Feel free to modify any part of the code to test new hypotheses and find better algorithms for unsupervised skill discovery. We offer two pieces of advice:
1. When doing hyperparameter searches, include the names of the hyperparameters in the list `TAG_KEYS`. Names included in `TAG_KEYS` will be appended to the checkpoint filenames.
2. Create a dedicated `logs` folder and number your experiments. In our experiments, we used `log_dir=<log_dir>/DIAYN_XXX`, where `XXX` was the experiment number.

# Open Problems

There are many open problems raised by DIAYN. Below, we offer a list of open problems. If interested in working on any of these, feel free to reach out to <eysenbach@google.com>.

### Easy
1. Try running DIAYN on a new environment. Some good candidates are the environments in [roboschool](https://github.com/openai/roboschool) and the [Box2D](https://github.com/openai/gym/tree/master/gym/envs/box2d) environments in Gym, and the environments in [pybullet](https://github.com/bulletphysics/bullet3). Send GIFs of trained skills to <eysenbach@google.com> for inclusion in the project website (we ensure you get credited).
2. Use pretrained skills to create a robot that dances to some song.
3. Use pretrained skills to create a game, where the human player decides which skill to use.

  ### Medium
4. There is a chicken-and-egg dilemma in DIAYN: skills learn to be diverse by using the discriminator's decision function, but the discriminator cannot learn to discriminate skills if they are not diverse. We found that synchronous updates of the policy and discriminator worked well enough for our experiments. We expect that more careful balancing of discriminator and policy updates would accelerate learning, leading to more diverse states. A first step in this direction is to modify the training loop to do N discriminator updates for every M policy updates, where N and M are tuneable hyperparameters.
5. While DIAYN was developed on top of [Soft Actor Critic](https://arxiv.org/abs/1801.01290), it could be applied on top of any RL algorithm. What are the benefits and limitations of applying DIAYN on top of (say) [PPO](https://arxiv.org/abs/1707.06347) and [Evolutionary Strategies](https://arxiv.org/abs/1703.03864)?
6. In preliminary experiments, we tried using a continuous prior distribution p(z). Neither a uniform nor Gaussian prior distribution allowed us to learn skills as well as a Categorical prior. Can continuous priors be made to work? One initial idea is to use a [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) as the prior, as the Dirichlet can interpolate between a categorical distribution (which we know works) and a uniform distribution (which doesn't work yet).

  ### Hard
7. Can skills discovered by used for hierarchical reinforcement learning? We suspect that answer is yes, but have not yet gotten it working. A good first step would be to learn a meta-policy that chooses which skill to use for the next (say) 100 steps.
8. Can we "evolve" an expanding set of diverse skills? In our experiments, we fixed the number of skills apriori, and sampled uniformly from this distribution during training. Taking inspiration from evolutionary algorithms, we might be able to learn a set of skills that expands over time. One approach is to start with two skills, and add a new skill every time the existing skills are sufficiently distinguishable.
9. Can new skills be initialized more intelligently? In our experiments, we randomly initialize a skill the first time it is used. An alternative would be to initialize the new skill with whatever existing skill is currently most distinguishable. This has the interpretation of doing a tree-search, where the most diverse leaf is split in two.
10. How can large scale, distributed training more quickly learn a large set of more diverse skills? A straightforward method for applying DIAYN in the distributed setting is to send a discriminator to each worker, and have each working independently learn a set of diverse skills. The skills are then aggregated by a central coordinator, which learns a new global discriminator. The new discriminator is sent to each worker, and the process is repeated.

# Credits
The DIAYN algorithm was developed by Benjamin Eysenbach, in collaboration with [Abhishek Gupta](https://people.eecs.berkeley.edu/~abhigupta/), [Julian Ibarz](https://research.google.com/pubs/JulianIbarz.html), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/). Work was completed while Benjamin was a member of the [Google AI Residency](https://research.google.com/teams/brain/residency/). We thank [Tuomas Haarnoja](https://people.eecs.berkeley.edu/~haarnoja/) for the implementation of [SAC](https://github.com/haarnoja/sac) on top of which this project is build.

# Legal
This project is released under the Apache 2.0 License. This is not an official Google product.

# Reference
```
@article{eysenbach2018,
    author  = "Eysenbach, Benjamin and Gupta, Abhishek and Ibarz, Julian and Levine, Sergey",
    title   = "Diversity is All You Need: Learning Diverse Skills without a Reward Function",
    year    = "2018"
}
```

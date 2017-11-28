import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def draw_rollout_path(rollout_path):
    observations = rollout_path["observations"]
    xy_positions = observations[:, :2]

    plt.plot(xy_positions[:, 0], xy_positions[:, 1])

def visitation_plot(rollout_paths):
    for rollout_path in rollout_paths:
        draw_rollout_path(rollout_path)
    plt.show()

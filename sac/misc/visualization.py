import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Wedge
import numpy as np
import re

FIG_SCALE = 2.0

def draw_rollout_path(h, xy_positions, colors=None, ptype='normal'):
    if ptype == 'scatter':
        plt.scatter(xy_positions[:, 0],
                    xy_positions[:, 1],
                    s=0.1,
                    c=colors)
    else:
        plt.plot(xy_positions[:, 0],
                 xy_positions[:, 1])
    # plt.legend()

def plot_visitation(rollout_paths, save_path=None):
    for h, rollout_path in enumerate(rollout_paths):
        draw_rollout_path(h, rollout_path['observations'][:, -3:-1], ptype='normal')

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.clf()

def plot_visitations(runs, variant=None, suptitle=None, save_path=None):
    num_plots = len(runs)
    num_rows = np.ceil(np.sqrt(num_plots))
    num_cols = num_rows

    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=(6.4 * FIG_SCALE, 4.8 * FIG_SCALE))

    if suptitle is not None:
        plt.suptitle(str(suptitle))

    for i, (run_id, rollout_paths) in enumerate(runs.items(), 1):
        ax = plt.subplot(num_rows, num_cols, i)
        for h, rollout_path in enumerate(rollout_paths):
            draw_rollout_path(
                h, rollout_path['observations'][:, -3:-1], ptype='normal')
            title = None
            plt.title(title, fontsize=22)
            # ax.axis('equal')
            ax.set_ylim(-30, 30)
            ax.set_xlim(-30, 30)

        theta1, theta2 = -45, 45
        plt.grid()
        # wedge = Wedge((0, 0), 3, theta1, theta2, fill=False, lw=2, zorder=1000)
        # wedge = Circle
        # ax.add_patch(wedge)

    # save_path = './visitation_test.pdf'
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def plot_hierarchy_visitations(data, variant, save_path):
    num_plots = len(data)
    num_rows = np.ceil(np.sqrt(num_plots))
    num_cols = num_rows

    fig, ax = plt.subplots(figsize=(6.4 * FIG_SCALE, 6.4 * FIG_SCALE))

    goal_radius = variant['env_goal_radius']
    # exp_name = variant['exp_name']

    for i, rollout_paths in enumerate(data, 1):
        ax = plt.subplot(num_rows, num_cols, i)
        ax.set_xlim(-5, 15)
        ax.set_ylim(-10, 10)
        # ax.axis('equal')

        goal_position = rollout_paths[0]['env_infos']['goal_position'][-1]

        goal_circle = plt.Circle(goal_position, goal_radius, color='g', label='GOAL', fill=False)
        ax.add_patch(goal_circle)
        plt.plot([0], [0], 'kx', label='START', markersize=10)
        plt.grid()

        for rollout_path in rollout_paths:
            assert np.all(
                rollout_path['env_infos']['goal_position'][-1] == goal_position)
            distances_from_goal = np.linalg.norm(
                rollout_path['observations'][:, -5:-3] - goal_position,
                axis=1)
            within_goal = distances_from_goal < goal_radius
            within_goal = rollout_path['rewards'] > 0
            # colors = np.array([
            #     'g' if x else 'r' for x in within_goal
            # ])

            # colors = np.zeros(shape=(within_goal.shape[0], 3), dtype=np.float32)
            # colors[np.where(within_goal)] = [0, 0, 0]
            # colors[np.where(~within_goal)] = [255, 0, 0]
            # draw_rollout_path(i, rollout_path['observations'][:, -5:-3], colors=colors)
            draw_rollout_path(i, rollout_path['observations'][:, -5:-3])

    plt.legend()
    plt.suptitle(
        "{}; goal_reward_weight={}"
        "".format(
            '/'.join(variant['low_level_policy_path'].split('/')[-2:]),
            variant['env_goal_reward_weight']))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

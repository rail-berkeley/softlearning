import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def draw_rollout_path(h, rollout_path):
    observations = rollout_path["observations"]
    xy_positions = observations[:, :2]

    plt.plot(xy_positions[:, 0],
             xy_positions[:, 1],
             label="h={}".format(h))
    # plt.legend()

def plot_visitation(rollout_paths, save_path=None):
    for h, rollout_path in enumerate(rollout_paths):
        draw_rollout_path(h, rollout_path)

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.clf()

def plot_visitations(runs, suptitle=None, save_path=None):
    num_plots = len(runs)
    num_rows = np.ceil(np.sqrt(num_plots))
    num_cols = num_rows

    if suptitle is not None:
        plt.suptitle(str(suptitle))

    for i, (run_id, rollout_paths) in enumerate(runs.items(), 1):
        plt.subplot(num_rows, num_cols, i)
        for h, rollout_path in enumerate(rollout_paths):
            draw_rollout_path(h, rollout_path)
            plt.title("run_id={}".format(run_id))

    plt.show()

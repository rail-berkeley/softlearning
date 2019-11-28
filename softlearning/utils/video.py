import os

import numpy as np


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(video_frames, filename, fps=60, video_format='mp4'):
    assert fps == int(fps), fps
    import skvideo.io
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )


def create_video_grid(col_and_row_frames):
    video_grid_frames = np.concatenate([
        np.concatenate(row_frames, axis=-2)
        for row_frames in col_and_row_frames
    ], axis=-3)

    return video_grid_frames

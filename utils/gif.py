"""Requirement: imagemagick"""

from matplotlib import animation
import matplotlib.pyplot as plt
import gym

from typing import Sequence

def save_frames(
        frames: Sequence,
        path: str,
        h: float
):
    """ Borrowed from here:
        https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=h*1000.)
    anim.save(path, writer='imagemagick')

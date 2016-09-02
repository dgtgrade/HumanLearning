import matplotlib.pyplot as plt
from mnist2ndarray import *
import time

print ("loading training images...")
train_images = mnist2ndarray("data/train-images-idx3-ubyte")
print ("loading training labels...")
train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")
assert len(train_images) == len(train_labels)
m = len(train_images)
print ("loaded %d images" % m)

IMGSIZE=28

def fig_init():

    global fig, ax_i, ax_c, ax_n
    
    fig = plt.figure()

    ax_i = plt.subplot2grid((2,2), (0,0))
    ax_c = plt.subplot2grid((2,2), (0,1))
    ax_n = plt.subplot2grid((2,2), (1,0))

    ax_i.set_position([0.15,0.2,0.7,0.7])
    ax_c.set_position([0.85,0.3,0.05,0.4])
    ax_n.set_position([0.15,0.05,0.7,0.1])

def fig_draw(i):

    ax_i.cla()
    ax_i.xaxis.set_ticks(np.arange(IMGSIZE))
    ax_i.xaxis.set_ticks(np.arange(IMGSIZE)+0.5, minor=True)
    ax_i.yaxis.set_ticks(np.arange(IMGSIZE)) 
    ax_i.yaxis.set_ticks(np.arange(IMGSIZE)+0.5, minor=True)
    [label.set_visible(False) for label in ax_i.get_xticklabels()] 
    [label.set_visible(False) for label in ax_i.get_yticklabels()]
    ax_i.grid(True, which='minor')
    ax_c.cla()
    ax_n.cla()
    ax_n.set_axis_off()
    im = ax_i.imshow(train_images[i], interpolation='none',
            cmap="gray_r", vmin=0, vmax=255)
    cb = fig.colorbar(im, cax=ax_c, orientation='vertical')
    cb.set_ticks([0,127,255], update_ticks=True)
    ax_n.text(0.5, 0.5, "[%s]" % train_labels[i],
            verticalalignment='center', horizontalalignment='center',
            transform=ax_n.transAxes, fontsize=50)


fig_init()

for i in range(m):
    fig_draw(i)
    plt.show(block=False)
    plt.pause(1)


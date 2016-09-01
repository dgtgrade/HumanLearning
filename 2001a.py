import matplotlib.pyplot as plt
from mnist2ndarray import *
import time

train_images = mnist2ndarray("data/train-images-idx3-ubyte")
train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")
assert len(train_images) == len(train_labels)
m = len(train_images)
print ("loaded %d images" % m)

fig = plt.figure()
ax_i = plt.subplot2grid((2,2), (0,0))
ax_c = plt.subplot2grid((2,2), (0,1))
ax_n = plt.subplot2grid((2,2), (1,0))

ax_i.set_position([0.15,0.2,0.7,0.7])
ax_c.set_position([0.85,0.3,0.05,0.4])
ax_n.set_position([0.15,0.05,0.7,0.1])

for i in range(m):
    ax_i.cla()
    ax_i.xaxis.tick_top()
    ax_c.cla()
    ax_n.cla()
    ax_n.set_axis_off()
    im = ax_i.imshow(train_images[i], cmap="gray_r", vmin=0, vmax=255)
    cb = fig.colorbar(im, cax=ax_c, orientation='vertical')
    cb.set_ticks([0,127,255], update_ticks=True)
    ax_n.text(0.5, 0.5, "[%s]" % train_labels[i],
            verticalalignment='center', horizontalalignment='center',
            transform=ax_n.transAxes, fontsize=50)
    plt.show(block=False)
    plt.pause(1)


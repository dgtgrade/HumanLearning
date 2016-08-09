#!/usr/bin/python

# targets (x, y) pairs

import matplotlib.pyplot as plt

targets_X = [row[0] for row in targets]
targets_Y = [row[1] for row in targets]

plt.plot(targets_X, targets_Y, "o")
plt.axis([0,10,0,10])
plt.show()


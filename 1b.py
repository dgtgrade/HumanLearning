#!/usr/bin/python

# trainings (x, y) pairs

import matplotlib.pyplot as plt

trainings = [[1,2],[3,6],[7,14],[2,4],[4,8]]

trainings_X = [row[0] for row in trainings]
trainings_Y = [row[1] for row in trainings]

plt.plot(trainings_X, trainings_Y, "o")
plt.axis([0,20,0,20])
plt.show()


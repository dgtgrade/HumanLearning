#!/usr/bin/python

import matplotlib.pyplot as plt
import random
import time

# targets (x, y) pairs
targets = [[1,2],[2,3],[3,6],[4,5],[5,8]]

targets_X = [row[0] for row in targets]
targets_Y = [row[1] for row in targets]

# 가정: y = ax

f, (ax1, ax2) = plt.subplots(1,2)
plt.ion()
plt.show()

a_min = 0 
a_max = 5 
a_step = 0.01

a = a_min

ax2.axis([0,5,0,100])

while a <= a_max:

    err = 0

    ax1.axis([-10,10,-10,10])
    ax1.plot(targets_X, targets_Y, marker="o", color="blue", linestyle="None")

    # 오차
    for i, t_x in enumerate(targets_X):

        y = a * t_x # 가정값

        t_y = targets_Y[i] # 실제값

        err += (y - t_y)**2 # 오차값 Squared Error

        ax1.plot(t_x, y, marker="D", color="green")

    mse = err / len(targets_X)

    errstr = "a="+str(int(a*100)/100)+"\n"+"err="+str(int(err*100)/100)

    ax1.text(-8,5,errstr,size=24)

    ax2.plot(a, err, marker="x", color="red")

    plt.pause(0.01)
    plt.draw()

    ax1.cla()

    a += a_step


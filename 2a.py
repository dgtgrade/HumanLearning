#!/usr/bin/python

import matplotlib.pyplot as plt
import random
import time

# targets (x, y) pairs
targets = [[1,2],[2,3],[3,6],[4,5],[5,8]]

targets_X = [row[0] for row in targets]
targets_Y = [row[1] for row in targets]

# 가정: y = ax

plt.ion()
plt.show()

a_min = 0
a_max = 5 
a_step = 0.1

a = a_min

while a <= a_max:

    #a = random.random()
    #a = random.uniform(-10,10)

    err = 0

    plt.axis([-10,10,-10,10])
    plt.plot(targets_X, targets_Y, marker="o", color="blue", linestyle="None")

    # 오차
    for i, t_x in enumerate(targets_X):

        y = a * t_x # 가정값

        t_y = targets_Y[i] # 실제값

        err += abs(y - t_y) # 오차값

        plt.plot(t_x, y, marker="D", color="green")

    errstr = "a="+str(int(a*100)/100)+"\n"+"err="+str(int(err*100)/100)
    plt.text(-8,5,errstr,size=24)
    plt.draw()
    plt.waitforbuttonpress(timeout=-1)
    plt.clf()

    a += a_step


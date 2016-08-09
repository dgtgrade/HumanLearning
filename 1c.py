#!/usr/bin/python

import matplotlib.pyplot as plt

# targets (x, y) pairs
targets = [[1,1],[2,2],[3,3],[4,4],[5,5]]

targets_X = [row[0] for row in targets]
targets_Y = [row[1] for row in targets]

import random
import time

# 가정: y = ax

while True:

    a = random.random()

    err = 0

    # 오차
    for i, t_x in enumerate(targets_X):

        y = a * t_x # 가정값

        t_y = targets_Y[i] # 실제값

        err += abs(y - t_y) # 오차값

    print ("a=",a,"err=",err)

    time.sleep(1)




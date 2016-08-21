#!/usr/bin/python

import matplotlib.pyplot as plt

# trainings (x, y) pairs
trainings = [[1,2],[3,6],[7,14],[2,4],[4,8]]

trainings_X = [row[0] for row in trainings]
trainings_Y = [row[1] for row in trainings]

import random
import time

# 가정: y = ax + b

MAX = 10000
best_a = -MAX
best_b = -MAX
err_min = MAX^3
count = 0

while True:

    a = random.uniform(-MAX,MAX)
    b = random.uniform(-MAX,MAX)

    err = 0
    count = count + 1

    # 오차
    for i, t_x in enumerate(trainings_X):

        y = a * t_x + b # 가정값

        t_y = trainings_Y[i] # 실제값

        err += abs(y - t_y) # 오차값

    if err < err_min:
        err_min = err
        best_a = a
        best_b = b

    if count % 10000 == 0 :
        print ("###count=", count)
        print ("a=",a,"b=",b,"err=",err)
        print ("    ", "best_a=",best_a,"best_b=",best_b,"err_min=",err_min)

#    time.sleep(0.01)




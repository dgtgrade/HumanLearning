import math
import random
import time

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

activate = sigmoid

# Supervised Learning - Training Set
train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
train_y = [0, 1, 1, 0]

n = len(train_x)
r_max = 1
iter = 0

while True:

    iter += 1

    w3 = [random.uniform(-r_max, r_max), random.uniform(-r_max, r_max), random.uniform(-r_max, r_max)]
    w4 = [random.uniform(-r_max, r_max), random.uniform(-r_max, r_max), random.uniform(-r_max, r_max)]
    wo = [random.uniform(-r_max, r_max), random.uniform(-r_max, r_max), random.uniform(-r_max, r_max)]

    # w3 = [-0.71, -0.94, 0.49]
    # w4 = [-0.97, 0.96, -0.47]
    # wo = [-0.58, 0.92, 0.94]

    # w3 = [-0.44, -0.900, 0.721]
    # w4 = [-0.7, 0.96, -0.654]
    # wo = [-0.34, 0.4616, 0.4334]
    all_loss = 0
    correct_prediction = 0

    for i in range(n):

        x1 = train_x[i][0]
        x2 = train_x[i][1]

        z3 = 1 * w3[0] + x1 * w3[1] + x2 * w3[2]
        x3 = activate(z3)
        z4 = 1 * w4[0] + x1 * w4[1] + x2 * w4[2]
        x4 = activate(z4)

        zo = 1 * wo[0] + x3 * wo[1] + x4 * wo[2]
        xo = activate(zo)

        out = 1 if xo > 0.5 else 0
        out_true = train_y[i]

        if out == out_true:
            correct_prediction += 1

        loss = (out_true - xo) ** 2

        print("i:{}, xo:{:7.5f}, out:{}, out_true:{}, loss:{:7.5f}".format(
            i, xo, out, out_true, loss))
        all_loss += loss

    all_loss /= n
    correct_prediction_pct = correct_prediction / n * 100

    print("{}: {}% {}".format(iter, correct_prediction_pct, all_loss))

    # time.sleep(5)

    if correct_prediction == n:
        print(w3)
        print(w4)
        print(wo)
        break




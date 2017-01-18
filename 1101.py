import numpy as np

#
RAND_MAX = 10

# train set
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([[0], [1], [1], [0]])

# number of train examples
m = train_x.shape[0]

# number of nodes for each layers
n_input_layer = train_x.shape[1]
n_hidden_layer1 = 2
n_hidden_layer2 = 2
n_output_layer = train_y.shape[1]


def threshold(z: np.ndarray):
    return (z > 0).astype(np.float)

activate = threshold


def add_bias(a: np.ndarray):
    return np.append([1], a)


epoch = 0

while True:

    epoch += 1
    pred_y = np.empty(train_y.shape)
    loss = 0

    w0 = None
    w1 = None
    w2 = None

    for i in range(m):

        # input layer
        input_layer = train_x[i]

        # connections between input layer and hidden layer1
        w0 = np.random.random(size=(n_input_layer + 1, n_hidden_layer1))
        w0 = (w0 * 2 - 1) * RAND_MAX

        # hidden layer1
        hidden_layer1_z = np.dot(add_bias(input_layer), w0)
        hidden_layer1_a = activate(hidden_layer1_z)

        #
        w1 = np.random.random(size=(n_hidden_layer1 + 1, n_hidden_layer2))
        w1 = (w1 * 2 - 1) * RAND_MAX

        #
        hidden_layer2_z = np.dot(add_bias(hidden_layer1_z), w1)
        hidden_layer2_a = activate(hidden_layer2_z)

        #
        w2 = np.random.random(size=(n_hidden_layer2 + 1, n_output_layer))
        w2 = (w2 * 2 - 1) * RAND_MAX

        # output layer
        output_layer_z = np.dot(add_bias(hidden_layer2_a), w2)
        output_layer_a = activate(output_layer_z)

        pred_y[i] = output_layer_a

        print("epoch #: {}, example #: {}, x: {}, y_true: {}, y: {}".format(
            epoch, i, train_x[i], train_y[i], output_layer_a))

    correct_prediction = np.sum(pred_y == train_y) / m

    print("epoch #: {}, correct result: {:5.3f}%".format(
        epoch, correct_prediction * 100))

    if correct_prediction == 1.0:
        print("w0:")
        print(w0)
        print("w1:")
        print(w1)
        print("w2:")
        print(w2)
        break


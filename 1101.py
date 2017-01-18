import numpy as np

#
RAND_MAX = 1

# train set: XOR
# train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# train_y = np.array([[0], [1], [1], [0]])

# train set: unknown function
# train_x = np.array([0, 1, 2, 5, 9, 11, 15])
# train_y = np.array([0, 1, 0, 1, 1, 1, 0])

# train set: SIN function
train_x = np.array([0, 1, 2, 5, 9, 11, 15])
train_y = np.sin(train_x)

#
train_x = train_x.reshape(len(train_x), -1)
train_y = train_y.reshape(len(train_x), -1)

# number of train examples
m = train_x.shape[0]

# number of nodes for each layers
n_input_layer = train_x.shape[1]
n_hidden_layer = 30
n_output_layer = train_y.shape[1]


def threshold(z: np.ndarray):
    return (z > 0.5).astype(np.float)


def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


# activate = threshold
activate = sigmoid


def add_bias(a: np.ndarray):
    return np.append([1], a)


epoch = 0

while True:

    epoch += 1
    preds = np.empty(train_y.shape)
    loss = 0

    w0 = None
    w1 = None
    w2 = None

    for i in range(m):

        # input layer
        input_layer = train_x[i]

        # connections between input layer and hidden layer
        w0 = np.random.random(size=(n_input_layer + 1, n_hidden_layer))
        w0 = (w0 * 2 - 1) * RAND_MAX

        # hidden layer
        hidden_layer_z = np.dot(add_bias(input_layer), w0)
        hidden_layer_a = activate(hidden_layer_z)

        # connections between hidden layer and output layer
        w1 = np.random.random(size=(n_hidden_layer + 1, n_output_layer))
        w1 = (w1 * 2 - 1) * RAND_MAX

        # output layer
        output_layer_z = np.dot(add_bias(hidden_layer_a), w1)
        output_layer_a = activate(output_layer_z)

        # y_pred = threshold(output_layer_a)
        y_pred = np.array(output_layer_a)

        #
        y_true = train_y[i]

        #
        loss += np.sum((y_pred - y_true) ** 2)
        preds[i] = y_pred

        print("epoch #: {}, example #: {}, x: {}, y_true: {}, y: {}".format(
            epoch, i, train_x[i], y_true, y_pred))

    correct_prediction = np.sum(preds == train_y) / m

    print("epoch #: {}, correct result: {:5.3f}%, loss: {:6.3f}".format(
        epoch, correct_prediction * 100, loss))

    if correct_prediction == 1.0:
        print("w0:")
        print(w0)
        print("w1:")
        print(w1)
        break


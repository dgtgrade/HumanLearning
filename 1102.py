import numpy as np

#
RAND_MAX = 10

#
PRINT_STATUS_PER_EPOCHS = 100


#
float_formatter = lambda x: "%+.6f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


# train set: XOR
# train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# train_y = np.array([[0], [1], [1], [0]])

# train set: unknown function
# train_x = np.array([0, 1, 2, 5, 7, 8, 9])
# train_y = np.array([0, 1, 0, 1, 1, 1, 0])

# train set: SIN function
train_x = np.array([0, 1, 2, 3, 6, 8])
train_y = np.sin(train_x)

#
train_x = train_x.reshape(len(train_x), -1)
train_y = train_y.reshape(len(train_x), -1)

# number of train examples
m = train_x.shape[0]

# number of nodes for each layers
n_input_layer = train_x.shape[1]
n_hidden_layer = 50
n_output_layer = train_y.shape[1]


def threshold(z: np.ndarray):
    return (z > 0.5).astype(np.float)


def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(a: np.ndarray):
    return a * (1.0 - a)

# activate = threshold
activate = sigmoid


def add_bias(a):
    return np.append([1], a)


def unroll(w0, w1):
    return np.append(w0.flatten(), w1.flatten())


def roll(w):
    w0 = w[0: (n_input_layer + 1) * n_hidden_layer].reshape((n_input_layer + 1), n_hidden_layer)
    w1 = w[(n_input_layer + 1) * n_hidden_layer:].reshape((n_hidden_layer + 1), n_output_layer)

    return w0, w1


def feed_forward(x, w):

    # input layer
    input_layer = x

    #
    w0, w1 = roll(w)

    # hidden layer
    hidden_layer_z = np.dot(add_bias(input_layer), w0)
    hidden_layer_a = activate(hidden_layer_z)

    # output layer
    output_layer_z = np.dot(add_bias(hidden_layer_a), w1)
    # output_layer_a = activate(output_layer_z)

    # out = threshold(output_layer_a)
    # out = np.array(output_layer_a)
    out = np.array(output_layer_z)

    return x, hidden_layer_a, out


learning_rate = 0.05


def num_grad_desc(x, w, y_true):

    delta = 0.000001

    tmp_w = w.copy()
    new_w = np.empty(w.shape)

    for i in range(len(new_w)):

        tmp_w[i] = w[i] + delta
        l2 = loss(feed_forward(x, tmp_w)[2], y_true)
        tmp_w[i] = w[i] - delta
        l1 = loss(feed_forward(x, tmp_w)[2], y_true)
        tmp_w[i] = w[i]

        new_w[i] = tmp_w[i] - learning_rate * (l2 - l1) / (delta * 2)

    return new_w


def back_propagation(x, w, y_true):

    tmp_w = w.copy()
    (tmp_w0, tmp_w1) = roll(tmp_w)

    a0, a1, a2 = feed_forward(x, w)
    a0 = add_bias(a0)
    a1 = add_bias(a1)

#    delta2 = (a2 - y_true)*d_sigmoid(a2)
    delta2 = (a2 - y_true)
    delta1 = (np.dot(tmp_w1, delta2) * d_sigmoid(a1))[1:]

    new_w1 = tmp_w1 - learning_rate * np.dot(a1.reshape(-1, 1), delta2.reshape(-1, 1).T)
    new_w0 = tmp_w0 - learning_rate * np.dot(a0.reshape(-1, 1), delta1.reshape(-1, 1).T)

    new_w = unroll(new_w0, new_w1)

    return new_w


def loss(y_pred: np.ndarray, y_true:np.ndarray):
    return np.sum((y_pred - y_true) ** 2) / 2

epoch = 0

# connections between input layer and hidden layer
my_w0 = np.random.random(size=(n_input_layer + 1, n_hidden_layer))
my_w0 = (my_w0 * 2 - 1) * RAND_MAX

# connections between hidden layer and output layer
my_w1 = np.random.random(size=(n_hidden_layer + 1, n_output_layer))
my_w1 = (my_w1 * 2 - 1) * RAND_MAX

while True:

    epoch += 1
    preds = np.empty(train_y.shape)
    total_loss = 0.0

    my_w = unroll(my_w0, my_w1)
    my_new_w = np.empty((m, unroll(my_w0, my_w1).size))

    for i in range(m):

        #
        my_x = train_x[i]
        my_y_true = train_y[i]

        #
        # my_new_w_ngd = num_grad_desc(my_x, my_w, my_y_true)
        my_new_w_bp = back_propagation(my_x, my_w, my_y_true)

        # print(my_new_w_ngd)
        # print(my_new_w_bp)
        # print(my_new_w_ngd - my_new_w_bp < 0.001)

        my_new_w[i] = my_new_w_bp

        #
        _, _, my_y_pred = feed_forward(train_x[i], my_w)

        #
        total_loss += loss(my_y_pred, my_y_true)
        preds[i] = my_y_pred

        if epoch % PRINT_STATUS_PER_EPOCHS == 0:
            print("epoch #: {}, example #: {}, x: {}, y_true: {}, y_pred: {}, y_diff: {}".format(
                epoch, i, train_x[i], my_y_true, my_y_pred, my_y_pred - my_y_true))

    my_w = np.average(my_new_w, axis=0)
    my_w0, my_w1 = roll(my_w)

    correct_prediction = np.array(np.abs(preds - train_y) < 0.01)
    correct_result = np.sum(correct_prediction) / m

    if epoch % PRINT_STATUS_PER_EPOCHS == 0:
        print("epoch #: {}, correct result: {:5.3f}%, loss: {:8.5f}".format(
            epoch, correct_result * 100, total_loss))

    if correct_result == 1.0:
        print("w0:")
        print(my_w0)
        print("w1:")
        print(my_w1)
        break


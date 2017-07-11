import numpy as np

#
RAND_MAX = 10

#
PRINT_STATUS_PER_EPOCHS = 100


#
float_formatter = lambda x: "%+.6f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})


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
n_hidden_layer_0 = 50
n_hidden_layer_1 = 50
n_output_layer = train_y.shape[1]


def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(a: np.ndarray):
    return a * (1.0 - a)


def ReLU(z: np.ndarray):
    return np.maximum(z, 0)


def d_ReLU(a: np.ndarray):
    return (a > 0).astype(np.float)


activate = sigmoid
d_activate = d_sigmoid


def add_bias(a):
    return np.append([1], a)


def unroll(w0, w1, w2):
    return np.concatenate((w0.flatten(), w1.flatten(), w2.flatten()))


def roll(w):

    n_w = np.array([[n_input_layer + 1, n_hidden_layer_0],
                    [n_hidden_layer_0 + 1, n_hidden_layer_1],
                    [n_hidden_layer_1 + 1, n_output_layer]])
    w0 = w[0: np.prod(n_w[0, :])].reshape(n_w[0, :])
    w1 = w[np.prod(n_w[0, :]): np.prod(n_w[0, :]) + np.prod(n_w[1, :])].reshape(n_w[1, :])
    w2 = w[np.prod(n_w[0, :]) + np.prod(n_w[1, :]):].reshape(n_w[2, :])

    return w0, w1, w2


def feed_forward(x, w):

    # input layer
    input_layer = x

    #
    w0, w1, w2 = roll(w)

    # hidden layer
    hidden_layer_0_z = np.dot(add_bias(input_layer), w0)
    hidden_layer_0_a = activate(hidden_layer_0_z)

    # hidden layer
    hidden_layer_1_z = np.dot(add_bias(hidden_layer_0_a), w1)
    hidden_layer_1_a = activate(hidden_layer_1_z)

    # output layer
    output_layer_z = np.dot(add_bias(hidden_layer_1_a), w2)
    # output_layer_a = activate(output_layer_z)

    # out = threshold(output_layer_a)
    # out = np.array(output_layer_a)
    out = np.array(output_layer_z)

    return x, hidden_layer_0_a, hidden_layer_1_a, out


learning_rate = 0.001


def num_grad_desc(x, w, y_true):

    delta = 0.000001

    tmp_w = w.copy()
    new_w = np.empty(w.shape)

    for i in range(len(new_w)):

        tmp_w[i] = w[i] + delta
        l2 = loss(feed_forward(x, tmp_w)[3], y_true)
        tmp_w[i] = w[i] - delta
        l1 = loss(feed_forward(x, tmp_w)[3], y_true)
        tmp_w[i] = w[i]

        new_w[i] = tmp_w[i] - learning_rate * (l2 - l1) / (delta * 2)

    return new_w


def back_propagation(x, w, y_true):

    tmp_w = w.copy()
    (tmp_w0, tmp_w1, tmp_w2) = roll(tmp_w)

    a0, a1, a2, a3 = feed_forward(x, w)
    a0 = add_bias(a0)
    a1 = add_bias(a1)
    a2 = add_bias(a2)

#    delta3 = (a3 - y_true)*d_activate(a3)
    delta3 = (a3 - y_true)
    new_w2 = tmp_w2 - learning_rate * np.dot(a2.reshape(-1, 1), delta3.reshape(-1, 1).T)

    delta2 = (np.dot(tmp_w2, delta3) * d_activate(a2))[1:]
    new_w1 = tmp_w1 - learning_rate * np.dot(a1.reshape(-1, 1), delta2.reshape(-1, 1).T)

    delta1 = (np.dot(tmp_w1, delta2) * d_activate(a1))[1:]
    new_w0 = tmp_w0 - learning_rate * np.dot(a0.reshape(-1, 1), delta1.reshape(-1, 1).T)

    new_w = unroll(new_w0, new_w1, new_w2)

    return new_w


def loss(y_pred: np.ndarray, y_true: np.ndarray):
    return np.sum((y_pred - y_true) ** 2) / 2

epoch = 0

# connections between layers
my_w0 = np.random.random(size=(n_input_layer + 1, n_hidden_layer_0))
my_w1 = np.random.random(size=(n_hidden_layer_0 + 1, n_hidden_layer_1))
my_w2 = np.random.random(size=(n_hidden_layer_1 + 1, n_output_layer))

my_w0 = (my_w0 * 2 - 1) * RAND_MAX
my_w1 = (my_w1 * 2 - 1) * RAND_MAX
my_w2 = (my_w2 * 2 - 1) * RAND_MAX

while True:

    epoch += 1
    preds = np.empty(train_y.shape)
    total_loss = 0.0

    my_w = unroll(my_w0, my_w1, my_w2)
    my_new_w = np.empty((m, unroll(my_w0, my_w1, my_w2).size))

    for my_i in range(m):

        #
        my_x = train_x[my_i]
        my_y_true = train_y[my_i]

        #
        # my_new_w_ngd = num_grad_desc(my_x, my_w, my_y_true)
        my_new_w_bp = back_propagation(my_x, my_w, my_y_true)

        # print(my_new_w_ngd)
        # print(my_new_w_bp)
        # print('number of different w\'s of ngd & bp: {}'.format(
        #     np.sum(np.array(my_new_w_ngd - my_new_w_bp > 0.001, dtype=int))))

        my_new_w[my_i] = my_new_w_bp

        #
        _x, _h_0, _h_1, my_y_pred = feed_forward(train_x[my_i], my_w)

        #
        total_loss += loss(my_y_pred, my_y_true)
        preds[my_i] = my_y_pred

        if epoch % PRINT_STATUS_PER_EPOCHS == 0:
            print("epoch #: {}, example #: {}, x: {}, y_true: {}, y_pred: {}, y_diff: {}".format(
                epoch, my_i, train_x[my_i], my_y_true, my_y_pred, my_y_pred - my_y_true))

    my_w = np.average(my_new_w, axis=0)
    my_w0, my_w1, my_w2 = roll(my_w)

    correct_prediction = np.array(np.abs(preds - train_y) < 0.01)
    correct_result = np.sum(correct_prediction) / m

    if correct_result == 1.0 or epoch % PRINT_STATUS_PER_EPOCHS == 0:
        print("epoch #: {}, correct result: {:5.3f}%, loss: {:8.5f}".format(
            epoch, correct_result * 100, total_loss))

    if correct_result == 1.0:
        print("w0:")
        print(my_w0)
        print("w1:")
        print(my_w1)
        print("w2:")
        print(my_w2)
        break



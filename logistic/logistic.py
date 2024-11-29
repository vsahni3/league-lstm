"""Logistic regression functions.

THIS CODE WAS TAKEN FROM CSC311 HW2.
"""
import numpy as np
Array = np.ndarray


def logistic_predict(weights: Array, data: Array) -> Array:
    """Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    # YOUR CODE BEGINS HERE
    data_with_bias = np.hstack((data, np.ones((data.shape[0], 1))))
    y = sigmoid(np.dot(data_with_bias, weights))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets: Array, y: Array) -> tuple[float, float]:
    """Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    # YOUR CODE BEGINS HERE

    # prevent numerical issues
    epsilon = 1e-15
    y_clamped = np.clip(y, epsilon, 1 - epsilon)

    flat_t = targets.flatten()
    flat_y = np.log(y_clamped).flatten()
    flat_minus_t = (1 - targets).flatten()
    flat_minus_y = np.log(1 - y_clamped).flatten()

    ce = (-np.dot(flat_t, flat_y) - np.dot(flat_minus_t, flat_minus_y)) / y.shape[0]
    frac_correct = float(np.mean(targets == np.round(y)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(
    weights: Array, data: Array, targets: Array
) -> tuple[float, Array, Array]:
    """Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :returns: A tuple (ce, frac_correct, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    # YOUR CODE BEGINS HERE
    difference = y - targets  # N x 1 vector
    n = data.shape[0]

    data_with_bias = np.hstack((data, np.ones((data.shape[0], 1))))

    f = evaluate(targets, y)[0]
    df = (1 / n) * np.dot(data_with_bias.transpose(), difference)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y

def sigmoid(x):
    """Computes the element wise logistic sigmoid of x."""
    return 1.0 / (1.0 + np.exp(-x))

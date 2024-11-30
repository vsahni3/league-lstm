import logistic_data as data
from check_grad import check_grad
from logistic import logistic, logistic_predict, evaluate


import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.02
WEIGHT_REGULARISATION = 0.0
NUM_ITERATIONS = 100


def run_check_grad():
    """Performs gradient check on logistic function.
    :return: None

    Note: THIS IS TAKEN FROM CSC311 HW2.
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic, weights, 0.001, data, targets)

    print("diff =", diff)


if __name__ == "__main__":
    run_check_grad()

    dataset, labels = data.load_json("logistic/match_info.json")
    train_inputs, val_inputs, test_inputs, train_targets, val_targets, test_targets = data.split_data(dataset, labels)

    D = dataset.shape[1]
    weights = np.zeros((D + 1, 1))

    ce, frac_correct = 0, 0
    train_entropies, val_metrics = [], []
    ts = []

    for t in range(NUM_ITERATIONS):
        ts.append(t)
        _, df, y = logistic(weights, train_inputs, train_targets.reshape((-1, 1)))
        ce, frac_correct = evaluate(train_targets, y)
        train_entropies.append(ce)
        val_metrics.append(evaluate(val_targets, logistic_predict(weights, val_inputs))[0])

        weights = weights - (LEARNING_RATE * df).reshape(-1, 1)

    print("TRAINING DATA: ")
    print(f"num_iterations={NUM_ITERATIONS}, "
          f"learning_rate={LEARNING_RATE}")
    print(f"frac_correct={frac_correct}, ce={ce}")

    y = logistic_predict(weights, val_inputs)
    ce, frac_correct = evaluate(val_targets, y)
    print("VALIDATION DATA: ")
    print(f"num_iterations={NUM_ITERATIONS}, "
          f"learning_rate={LEARNING_RATE}")
    print(f"frac_correct={frac_correct}, ce={ce}")

    y = logistic_predict(weights, test_inputs)
    ce, frac_correct = evaluate(test_targets, y)
    print("TEST DATA: ")
    print(f"num_iterations={NUM_ITERATIONS}, "
          f"learning_rate={LEARNING_RATE}")
    print(f"frac_correct={frac_correct}, ce={ce}")

    plt.plot(ts, train_entropies)
    plt.show()

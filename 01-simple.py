import os
import time
from random import random

from mlflow import log_metric, log_param, log_artifacts, set_tag


def main():
    # Log a parameter (key-value pair)
    set_tag("algo", "test_algo")
    log_param("learning_rate", random())

    # Log a metric; metrics can be updated throughout the run
    t = 0.2
    epochs = 25
    log_param("epochs", epochs)
    output = []
    for i in range(epochs):
        time.sleep(0.1)
        _value = random() * t + i / 25 * (1 - t)
        log_metric("accuracy", _value, step=i+1)
        _msg = f"Epoch {i + 1} of {epochs} - accuracy: {_value}"
        print(_msg)
        output.append(_msg)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        for line in output:
            f.write(line + "\n")
    log_artifacts("outputs")


if __name__ == "__main__":
    main()

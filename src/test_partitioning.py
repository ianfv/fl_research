import dataset
import numpy as np


def main():
    train_set, test_set = dataset.load_mnist()

    labels = np.array(train_set.targets)

    print(np.unique(labels))


if __name__ == "__main__":
    main()

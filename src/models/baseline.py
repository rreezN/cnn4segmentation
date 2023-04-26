import numpy as np
from sklearn.naive_bayes import GaussianNB


def standardiseTransform(data, mean, std):
    # Standardise data
    return (data - mean) / std


if __name__ == '__main__':
    X = np.load("../../data/processed/training.npy")
    y = np.load("../../data/processed/training_labels.npy")

    # Standardisation parameters
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Standardise train data
    X = standardiseTransform(X, mu, std)

    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

    test_x = np.load("../../data/processed/test.npy")
    test_y = np.load("../../data/processed/test_labels.npy")
    test_x = standardiseTransform(test_x, mu, std)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])

    baseline = GaussianNB()
    baseline.fit(X, y)
    predictions = baseline.predict(test_x)
    print(f"Accuracy: {np.sum(predictions == test_y)/len(test_y)}")

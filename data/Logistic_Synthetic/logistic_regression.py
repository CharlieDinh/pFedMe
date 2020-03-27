#!/usr/bin/env python
import numpy as np
import json
import random
import os


def logit(X, W):
    return 1 / (1 + np.exp(-np.dot(X, W)))


def generate_logistic_regression_data(num_users=100, kappa=10, dim=40, noise_ratio=0.05):
    # For consistent results
    np.random.seed(0)

    # Sanity check
    assert(kappa >= 1 and num_users > 0 and dim > 0)

    X_split = [[] for _ in range(num_users)]  # X for each user
    y_split = [[] for _ in range(num_users)]  # y for each user

    # Find users' sample sizes based on the power law (heterogeneity)
    samples_per_user = np.random.lognormal(4, 2, num_users).astype(int) + 50
    indices_per_user = np.insert(samples_per_user.cumsum(), 0, 0, 0)
    num_total_samples = indices_per_user[-1]

    # Each user's mean is drawn from N(0, 1) (non-i.i.d. data)
    mean_X = np.array([np.random.randn(dim) for _ in range(num_users)])

    # Covariance matrix for X
    Sigma = np.eye(dim)

    # L = 1, beta = LAMBDA
    LAMBDA = 100 if kappa == 1 else 1 / (kappa - 1)

    # Keep all users' inputs and labels in one array,
    # indexed according to indices_per_user.
    #   (e.g. X_total[indices_per_user[n]:indices_per_user[n+1], :] = X_n)
    #   (e.g. y_total[indices_per_user[n]:indices_per_user[n+1]] = y_n)
    X_total = np.zeros((num_total_samples, dim))
    y_total = np.zeros(num_total_samples)

    for n in range(num_users):
        # Generate data
        X_n = np.random.multivariate_normal(mean_X[n], Sigma, samples_per_user[n])
        X_total[indices_per_user[n]:indices_per_user[n+1], :] = X_n

    # Normalize all X's using LAMBDA
    norm = np.sqrt(np.linalg.norm(X_total.T.dot(X_total), 2) / num_total_samples)
    X_total /= norm + LAMBDA

    # Generate weights and labels
    W = np.random.rand(dim)
    y_total = logit(X_total, W)
    y_total = np.where(y_total > 0.5, 1, 0)

    # Apply noise: randomly flip some of y_n with probability noise_ratio
    noise = np.random.binomial(1, noise_ratio, num_total_samples)
    y_total = np.multiply(noise - y_total, noise) + np.multiply(y_total, 1 - noise)

    # Save each user's data separately
    for n in range(num_users):
        X_n = X_total[indices_per_user[n]:indices_per_user[n+1], :]
        y_n = y_total[indices_per_user[n]:indices_per_user[n+1]]
        X_split[n] = X_n.tolist()
        y_split[n] = y_n.tolist()

        # print("User {} has {} samples.".format(n, samples_per_user[n]))

    print("=" * 80)
    print("Generated synthetic data for logistic regression successfully.")
    print("Summary of the generated data:".format(kappa))
    print("    Total # users       : {}".format(num_users))
    print("    Input dimension     : {}".format(dim))
    print("    rho                 : {}".format(kappa))
    print("    Total # of samples  : {}".format(num_total_samples))
    print("    Minimum # of samples: {}".format(np.min(samples_per_user)))
    print("    Maximum # of samples: {}".format(np.max(samples_per_user)))
    print("=" * 80)

    return X_split, y_split


def save_total_data():
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    train_path = os.path.join("data", "train", "mytrain.json")
    test_path = os.path.join("data", "test", "mytest.json")
    for path in [os.path.join("data", "train"), os.path.join("data", "test")]:
        if not os.path.exists(path):
            os.makedirs(path)

    X, y = generate_logistic_regression_data(100, 2, 40, 0.05)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(100):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    
    print("=" * 80)
    print("Saved all users' data sucessfully.")
    print("    Train path:", os.path.join(os.curdir, train_path))
    print("    Test path :", os.path.join(os.curdir, test_path))
    print("=" * 80)


def save_data_by_user():
    train_path = os.path.join("data", "userstrain")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    test_path = os.path.join("data", "userstest")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    with open(os.path.join("data", "train", "mytrain.json"), "r") as f:
        test = json.load(f)

    for i in range(100):
        data = {}
        data['id'] = test['users'][i]
        data['X'] = test["user_data"][data['id']]['x']
        data['y'] = test["user_data"][data['id']]['y']
        data['num_samples'] = test["num_samples"][i]
        with open(os.path.join(train_path, data['id'] + ".json"), "w") as f:
            json.dump(data, f)

    with open(os.path.join("data", "test", "mytest.json"), "r") as f_train:
        test = json.load(f_train)

    for i in range(100):
        data = {}
        data['id'] = test['users'][i]
        data['X'] = test["user_data"][data['id']]['x']
        data['y'] = test["user_data"][data['id']]['y']
        data['num_samples'] = test["num_samples"][i]
        with open(os.path.join(test_path, data['id'] + ".json"), "w") as f_test:
            json.dump(data, f_test)
    
    print("=" * 80)
    print("Saved each user's data sucessfully.")
    print("    Train path: {}".format(os.path.join(os.curdir, train_path, "*.json")))
    print("    Test path : {}".format(os.path.join(os.curdir, test_path, "*.json")))
    print("=" * 80)


def main():
    save_total_data()
    save_data_by_user()


if __name__ == '__main__':
    main()

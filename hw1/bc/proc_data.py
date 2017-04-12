import argparse
import pickle
import numpy as np


def print_stat(data, title):
    obs, actions = data["observations"], data["actions"]
    print("%s Observations %s, mean: %s", title, str(obs.shape), str(np.mean(obs, axis=0)))
    print("%s Actions %s, mean: %s", title, str(actions.shape), str(np.mean(actions, axis=0)))


def split(data, params):
    """Split the dataset into train, val, and test"""
    obs, actions = data["observations"], data["actions"]
    assert len(obs) == len(actions), "obs and action mismatch!"

    n_total = len(obs)
    n_train, n_val = n_total * params["train_ratio"], n_total * params["val_ratio"]

    train_data = {
        "observations": obs[:n_train],
        "actions": actions[:n_train]
    }

    val_data = {
        "observations": obs[n_train:n_train+n_val],
        "actions": actions[n_train:n_train+n_val]
    }

    test_data = {
        "observations": obs[n_train + n_val:],
        "actions": actions[n_train + n_val:]
    }

    return train_data, val_data, test_data


def pre_proc(data, mean, std):
    """Normalize observations"""
    pass


def get_small_dataset(train_data):
    """Return a subset of the training data"""
    pass


def main(params):
    data = pickle.load(open(params["expert_data_file"], "rb"))

    print("Splitting dataset...")
    train, val, test = split(data, params)
    print_stat(train, "Training")
    print_stat(val, "Validation")
    print_stat(test, "Test")

    print("Normalizing observations...")
    obs_mean, obs_std = np.mean(train["observations"], axis=0), np.std(train["observations"], axis=0)
    pre_proc(train, obs_mean, obs_std)
    pre_proc(val, obs_mean, obs_std)
    pre_proc(test, obs_mean, obs_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data_file", type=str, default="../data/Hopper_data.pkl")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--small_data", type=int, default=100)
    args = vars(parser.parse_args())

    main(args)

import datetime
import imp
import json
import os
import time
from random import seed

import numpy as np
import qamlz
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from roc_excited_states import main as roc_main

script_path = os.path.dirname(os.path.realpath(__file__))


def make_results_json(fail_flag=""):
    filename = "%sanneal_results-%s.json" % (fail_flag, model.start_time)
    destdir = os.path.join(script_path, "qamlz_runs")
    filepath = os.path.join(destdir, filename)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    json.dump(model.anneal_results, open(filepath, "w"), indent=4)

    return filename


def make_mus_files(fail_flag=""):
    mus_filenames = []
    mus_destdir = None
    for k, v in model.mus_dict.items():
        mus_filename = fail_flag + k
        mus_filenames.append(mus_filename)
        mus_destdir = os.path.join(script_path, "mus/%s" % model.start_time)
        mus_filepath = os.path.join(mus_destdir, mus_filename)
        if not os.path.exists(mus_destdir):
            os.makedirs(mus_destdir)
        np.save(mus_filepath, v)

    return mus_destdir, mus_filenames


# sig = np.loadtxt('sig.csv')
# bkg = np.loadtxt('bkg.csv')
# # X_train = np.vstack((sig[:100, :], bkg[:100, :]))
# # y_train = np.concatenate((np.ones(100), -np.ones(100)))
# full_X_data = np.vstack((sig, bkg))
# full_y_data = np.concatenate((np.ones(np.size(sig, axis=0)), -np.ones(np.size(bkg, axis=0))))
# sss = StratifiedShuffleSplit(n_splits=8, test_size=0.5, random_state=0)
# for train_index, test_index in sss.split(full_X_data, full_y_data):
#     X_train, X_test = full_X_data[train_index], full_X_data[test_index]
#     y_train, y_test = full_y_data[train_index], full_y_data[test_index]
# # np.random.seed(0)
# rand_train_indeces = np.random.choice(np.size(X_train, axis=0), size=5000)
# rand_test_indeces = np.random.choice(np.size(X_test, axis=0), size=5000)

sig_X = np.loadtxt("sig.csv")
bkg_X = np.loadtxt("bkg.csv")
sig_y = np.ones(np.size(sig_X, axis=0))
bkg_y = -np.ones(np.size(bkg_X, axis=0))
sig_X_train, sig_X_test, sig_y_train, sig_y_test = train_test_split(
    sig_X, sig_y, test_size=0.5, shuffle=True, random_state=0
)
bkg_X_train, bkg_X_test, bkg_y_train, bkg_y_test = train_test_split(
    bkg_X, bkg_y, test_size=0.5, shuffle=True, random_state=0
)

X_train, y_train = np.vstack((sig_X_train, bkg_X_train)), np.concatenate(
    (sig_y_train, bkg_y_train)
)

url = "https://cloud.dwavesys.com/sapi/"
token = os.environ["USC_DWAVE_TOKEN"]
# token = os.environ["DWAVE_TOKEN"]

train_sizes = [100, 1000, 5000, 10000, 15000, 20000]
for train_size in train_sizes:
    print("train_size = %d" % train_size)
    np.random.seed(0)
    rand_train_indeces = np.random.choice(np.size(X_train, axis=0), size=train_size)
    env = qamlz.TrainEnv(
        X_train[rand_train_indeces],
        y_train[rand_train_indeces],
        url,
        token,
        fidelity=7,
        dwave_topology="chimera",
    )
    config = qamlz.ModelConfig()
    config.cutoff_percentile = 95
    config.encode_vars = None
    model = qamlz.Model(config, env)
    try:
        model.train()
    except Exception as e:
        print(e)
        make_results_json("FAIL")
        mus_dir, mus_filenames = make_mus_files("FAIL")
        raise e
    make_results_json()
    mus_dir, mus_filenames = make_mus_files()
    time.sleep(5)
    roc_main(
        sig=sig_X_test,
        bkg=bkg_X_test,
        mus_dir=mus_dir,
        mus_filenames=mus_filenames,
        train_size=train_size,
    )

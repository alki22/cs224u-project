import data_readers
import models
import numpy as np
import os
import pandas as pd
import random

from collections import Counter
from config import Config
from console import log_info
from data_readers import read_dataset_splits, read_corpus
from dotdict import DotDict
from functools import reduce
from model_utils import get_response_time_label, add_cosine_similarity, add_question_length, add_jensen_shannon, plot_cm, dummy_tokenizer
from pathlib import Path
from progressbar import progressbar
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from sklearn.model_selection import ParameterSampler

random.seed(Config.SEED)

def prepare_data(data):
    y = data.apply(lambda x: 1 if '?' in x.response else 0, axis=1)
    X = data.drop(columns="response_time_sec").to_dict(orient="list")
    return X, y

class SklearnTrainer(object):
    def __init__(self, model, data_name,  n_samples):
        self.pipe = model.pipe
        self.directory = Path(os.path.join(Config.RUNS_DIR, data_name, model.name))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models_directory = Path(os.path.join(Config.MODELS_DIR, "cr_classifier", model.name))
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.params_sampler = ParameterSampler(model.params_range, n_iter=n_samples, random_state=Config.SEED)

    def train(self, train_data, dev_data):
        X_train, y_train  = prepare_data(train_data)
        X_dev, y_dev  = prepare_data(dev_data)
        self.best_clf = None
        self.best_params = None
        best_f1 = 0

        for params in progressbar(self.params_sampler):
            clf = self.pipe.set_params(**params)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_dev)
            f1 = f1_score(y_dev, preds, average='weighted')
            print("\tTrain F1: %.2f" % f1_score(y_train, clf.predict(X_train), average='weighted'))
            print("\tDev F1: %.2f" % f1)
            if f1 > best_f1:
                best_f1 = f1
                self.best_clf = clf
                self.best_params = params

        joblib.dump(self.best_clf, self.models_directory.joinpath('clf.sav'))
        self.eval(X_train, y_train, split="train")
        self.eval(X_dev, y_dev, split="test")

    def eval(self, X, y, split="tiny"):
        assert self.best_clf is not None

        split_dir = self.directory.joinpath(split)
        split_dir.mkdir(parents=True, exist_ok=True)

        preds = self.best_clf.predict(X)
        precision, recall, f1, support = precision_recall_fscore_support(y, preds, average='weighted')
        with split_dir.joinpath("results").open(mode='w') as results_file:
            print("Precision: %.4f" % (precision if precision is not None else -1), file=results_file)
            print("Recall: %.4f" % (recall if recall is not None else -1), file=results_file)
            print("F1-score: %.4f" % (f1 if f1 is not None else -1), file=results_file)
            print("Support: %d" % (support if support is not None else -1), file=results_file)

        report = classification_report(y, preds)
        with split_dir.joinpath("classification_report").open(mode='w') as report_file:
            print(report, file=report_file)

        cm = confusion_matrix(y, preds)
        with split_dir.joinpath("confusion_matrix").open(mode='w') as cm_file:
            np.savetxt(cm_file, cm, fmt="%d")
        
        plot_cm(cm, os.path.join(split_dir, "cm.png"))
        
        with split_dir.joinpath("params").open(mode='w') as params_file:
            print(self.best_params, file=params_file)   

if __name__ == '__main__':
    data = read_dataset_splits(reader=data_readers.read_question_and_response_data)

    print(type(data))
    print(data)

    trainer = SklearnTrainer(models.SVM, data_name="cr_classifier", n_samples=5)
    trainer.train(data.train, data.test)

    trainer2 = SklearnTrainer(models.Logistic, data_name="cr_classifier", n_samples=5)
    trainer2.train(data.train, data.test)


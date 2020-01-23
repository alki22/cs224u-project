import ast
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
from imblearn.under_sampling import RandomUnderSampler
from model_utils import get_response_time_label, add_cosine_similarity, add_question_length, add_jensen_shannon, plot_cm, dummy_tokenizer
from pathlib import Path
from progressbar import progressbar
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from sklearn.model_selection import ParameterSampler, train_test_split

random.seed(Config.SEED)

def read_dataset(filename):
    dtypes = {"response_time_sec": np.int32, "session_id": np.int32}
    converters = {"question": ast.literal_eval, "response": ast.literal_eval}
    path = filename
    data = pd.read_csv(path, sep=",", header=0, dtype=dtypes, converters=converters)
    return data

class SklearnTrainer(object):
    def __init__(self, model, data_name,  n_samples):
        self.pipe = model.pipe
        self.directory = Path(os.path.join(Config.RUNS_DIR, data_name, model.name))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models_directory = Path(os.path.join(Config.MODELS_DIR, "cr_classifier", model.name))
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.params_sampler = ParameterSampler(model.params_range, n_iter=n_samples, random_state=Config.SEED)

    def train(self):
        dataset_path = '../data/'

        train_df = read_dataset(dataset_path + 'train_question_text_and_response_text_dataset.csv')
        test_df = read_dataset(dataset_path + 'test_question_text_and_response_text_dataset.csv')
        dev_df = read_dataset(dataset_path + 'dev_question_text_and_response_text_dataset.csv')
        tiny_df = read_dataset(dataset_path + 'tiny_question_text_and_response_text_dataset.csv')

        data = pd.concat([train_df,test_df,dev_df,tiny_df])
        data = data.reset_index(drop=True)

        y = data.apply(lambda x: 1 if '?' in x.response else 0, axis=1)
        X = data.drop(columns="response_time_sec")

        self.best_clf = None
        self.best_params = None
        best_f1 = 0

        #Balance dataset and split
        random_under_sampler = RandomUnderSampler()
        X, y = random_under_sampler.fit_resample(X, y)
        print('Resampled dataset shape %s' % Counter(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=Config.SEED)


        X_train.to_dict(orient="list")
        X_test.to_dict(orient="list")

        for params in progressbar(self.params_sampler):
            clf = self.pipe.set_params(**params)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            f1 = f1_score(y_test, preds, average='weighted')
            print("\tTrain F1: %.2f" % f1_score(y_train, clf.predict(X_train), average='weighted'))
            print("\tDev F1: %.2f" % f1)
            if f1 > best_f1:
                best_f1 = f1
                self.best_clf = clf
                self.best_params = params

        joblib.dump(self.best_clf, self.models_directory.joinpath('clf.sav'))
        self.eval(X_train, y_train, split="train")
        self.eval(X_test, y_test, split="test")

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

    trainer = SklearnTrainer(models.SVM, data_name="cr_classifier", n_samples=5)
    trainer.train()

    trainer2 = SklearnTrainer(models.Logistic, data_name="cr_classifier", n_samples=5)
    trainer2.train()


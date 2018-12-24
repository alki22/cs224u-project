import data_readers
import models
import numpy as np
import os
import pandas as pd
import random

from collections import Counter
from config import Config
from data_readers import read_dataset_splits, read_corpus
from dotdict import DotDict
from functools import reduce
from model_utils import get_response_time_label, add_cosine_similarity, add_question_length, add_jensen_shannon, plot_cm
from pathlib import Path
from progressbar import progressbar
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from sklearn.model_selection import ParameterSampler

random.seed(Config.SEED)

def prepare_data(data):
    y = data.response_time_sec.apply(get_response_time_label).values
    X = data.drop(columns="response_time_sec").to_dict(orient="list")
    return X, y

class SklearnTrainer(object):
    def __init__(self, model, data_name,  n_samples):
        self.pipe = model.pipe
        self.directory = Path(os.path.join(Config.RUNS_DIR, data_name, model.name))
        self.directory.mkdir(parents=True, exist_ok=True)
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

        joblib.dump(self.best_clf, self.directory.joinpath('clf.pkl'))
        self.eval(X_train, y_train, split="train")
        self.eval(X_dev, y_dev, split="dev")

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
    texts = ['response']

    # Question text + Response Text (A) Models

    print("SVM Q+A")
    model = models.MultiTextSVM(texts)
    trainer = SklearnTrainer(model, data_name="question_and_response", n_samples=5)
    trainer.train(data.train, data.dev)
    
    print("LR Q+A")
    model = models.MultiTextLogistic(texts)
    trainer = SklearnTrainer(model, data_name="question_and_response", n_samples=5)
    trainer.train(data.train, data.dev)

    
    data = DotDict()
    
    q_length = read_dataset_splits(reader=data_readers.read_question_only_data)
    q_length = add_question_length(q_length)
    q_duration = read_dataset_splits(reader=data_readers.read_question_and_duration_data)
    q_context = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=5, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=True)
    
    for key in Config.SPLITS:
        arr = [q_length[key], q_duration[key], q_context[key]]
        data[key] = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), arr)
    
    window_size = 5
    texts = ["turn_text-%d" % i for i in range(1, window_size+1)]    
    scalars = ["turn_time-%d" % i for i in range(1, window_size+1)]
    scalars += ["question_length", "question_duration_sec"]
    
    # Question text + Question duration, length, previous turns text and times (Q+T+L+D+X)

    print("SVM Q+T+L+D+X")
    model = models.MultiTextSVMWithScalars(texts, scalars)
    trainer = SklearnTrainer(model, data_name="combined_length_duration_ctime_ctext", n_samples=5)
    trainer.train(data.train, data.dev)

    print("LR Q+T+L+D+X")
    model = models.MultiTextLogisticWithScalars(texts, scalars)
    trainer = SklearnTrainer(model, data_name="combined_length_duration_ctime_ctext", n_samples=5)
    trainer.train(data.train, data.dev)
    
    # Doubtful about whether it is Q+D or not because of the ctime on its data name

    """
    print("SVM Q+D")
    model = models.SVMWithScalars(scalars)
    trainer = SklearnTrainer(model, data_name="combined_length_duration_ctime", n_samples=5)
    trainer.train(data.train, data.dev)
    
    print("LR Q+D")
    model = models.LogisticWithScalars(scalars)
    trainer = SklearnTrainer(model, data_name="combined_length_duration_ctime", n_samples=5)
    trainer.train(data.train, data.dev)
    """

    scalars = ["turn_time-%d" % i for i in range(1, window_size+1)]
    scalars += ["question_length"]
    
    """
    trainer = SklearnTrainer(model, data_name="combined_length_ctime", n_samples=5)
    trainer.train(data.train, data.dev)
    
    model = models.LogisticWithScalars(scalars)
    trainer = SklearnTrainer(model, data_name="combined_length_ctime", n_samples=5)
    trainer.train(data.train, data.dev)
    """

    
    # Label Count Models

    data = read_dataset_splits(reader=data_readers.read_label_counts_data)

    print("Label counts SVM")
    model = models.SVMVector("label_counts")
    trainer = SklearnTrainer(model, data_name="label_counts", n_samples=5)
    trainer.train(data.train, data.dev)

    print("Label counts LR")
    model = models.LogisticVector("label_counts")
    trainer = SklearnTrainer(model, data_name="label_counts", n_samples=5)
    trainer.train(data.train, data.dev)

        
    # Question text + Previous dialogue turns text (X) with window size of 5
    
    data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=5, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)

    for window_size in [5]:
        texts = ["turn_text-%d" % i for i in range(1, window_size+1)]
        
        print("SVM Q+X with window size of %d" % window_size)
        model = models.MultiTextSVM(texts)
        trainer = SklearnTrainer(model, data_name="question_and_context_text_%d" % window_size, n_samples=5)
        trainer.train(data.train, data.dev)
        

        print("LR Q+X with window size of %d" % window_size)
        model = models.MultiTextLogistic(texts)
        trainer = SklearnTrainer(model, data_name="question_and_context_text_%d" % window_size, n_samples=5)
        trainer.train(data.train, data.dev)

    
    data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=5, include_question_text=True, include_context_text=False, include_context_speaker=False, include_context_times=True)
    
    # Question text + Previous dialogue turns times (T) with window size of 5
    
    for window_size in [5]:
        times = ["turn_time-%d" % i for i in range(1, window_size+1)]
        
        print("SVM Q+T with window size of %d" % window_size)
        model = models.SVMWithScalars(times)
        trainer = SklearnTrainer(model, data_name="question_and_context_time_%d" % window_size, n_samples=5)
        trainer.train(data.train, data.dev)
        
        print("LR Q+T with window size of %d" % window_size)
        model = models.LogisticWithScalars(times)
        trainer = SklearnTrainer(model, data_name="question_and_context_time_%d" % window_size, n_samples=5)
        trainer.train(data.train, data.dev)


    # Commented since we use the 25, 50 and 100 most common words in the paper
    """
    data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
    data = add_jensen_shannon(data)
    
    print("SVM Q+E using Jensen-Shannon divergence")
    trainer = SklearnTrainer(models.LogisticWithScalar("jensen_shannon"), data_name="question_and_js", n_samples=5)
    trainer.train(data.train, data.dev)
    
    print("LR Q+E using Jensen-Shannon divergence")
    trainer = SklearnTrainer(models.SVMWithScalar("jensen_shannon"), data_name="question_and_js", n_samples=5)
    trainer.train(data.train, data.dev)
    """
    
    df = read_corpus(split='train')
    all_words = [item for sublist in df.text for item in sublist]
    for N_words in [25, 50, 100]:
        top_words = [item[0] for item in Counter(all_words).most_common(N_words)]
        data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
        data = add_jensen_shannon(data, stopwords=top_words)
        print("LR Q+E using Jensen-Shannon divergence with %d most common words" % N_words)
        trainer = SklearnTrainer(models.LogisticWithScalar("jensen_shannon"), data_name="question_and_js_top" + str(N_words), n_samples=5)
        trainer.train(data.train, data.dev)
        print("SVM Q+E using Jensen-Shannon divergence with %d most common words" % N_words)
        trainer = SklearnTrainer(models.SVMWithScalar("jensen_shannon"), data_name="question_and_js_top" + str(N_words), n_samples=5)
        trainer.train(data.train, data.dev)
        
    #data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
    #data = add_cosine_similarity(data)
    #trainer = SklearnTrainer(models.LogisticWithScalar("cosine_similarity"), data_name="question_and_similarity", n_samples=5)
    #trainer.train(data.train, data.dev)
    #trainer = SklearnTrainer(models.SVMWithScalar("cosine_similarity"), data_name="question_and_similarity", n_samples=5)
    #trainer.train(data.train, data.dev)
    
    
    df = read_corpus(split='train')
    all_words = [item for sublist in df.text for item in sublist]
    for N_words in [25, 50, 100]:
        top_words = [item[0] for item in Counter(all_words).most_common(N_words)]
        data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=10, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
        data = add_cosine_similarity(data, stopwords=top_words)
        print("LR Q+E using cosine similarity with %d most common words" % N_words)
        trainer = SklearnTrainer(models.LogisticWithScalar("cosine_similarity"), data_name="question_and_similarity_top" + str(N_words), n_samples=5)
        trainer.train(data.train, data.dev)
        print("SVM Q+E using cosine similarity with %d most common words" % N_words)
        trainer = SklearnTrainer(models.SVMWithScalar("cosine_similarity"), data_name="question_and_similarity_top" + str(N_words), n_samples=5)
        trainer.train(data.train, data.dev)

    # Question text + Question length (L)

    data = read_dataset_splits(reader=data_readers.read_question_only_data)
    data = add_question_length(data)

    print("LR Q+L")
    trainer = SklearnTrainer(models.LogisticWithScalar("question_length"), data_name="question_and_length", n_samples=5)
    trainer.train(data.train, data.dev)
    
    print("SVM Q+L")
    trainer = SklearnTrainer(models.SVMWithScalar("question_length"), data_name="question_and_length", n_samples=5)
    trainer.train(data.train, data.dev)
    
    # Question text + Sentiment (S)

    data = read_dataset_splits(reader=data_readers.read_question_and_sentiment_data)
    
    print("LR Q+S")
    trainer = SklearnTrainer(models.LogisticWithScalar("question_sentiment"), data_name="question_and_sentiment", n_samples=5)
    trainer.train(data.train, data.dev)
    
    print("SVM Q+S")
    trainer = SklearnTrainer(models.SVMWithScalar("question_sentiment"), data_name="question_and_sentiment", n_samples=5)
    trainer.train(data.train, data.dev)
    

    #data = read_dataset_splits(reader=data_readers.read_question_and_newlines_data)
    #trainer = SklearnTrainer(models.SVM, data_name="question_and_newlines", n_samples=5)
    #trainer.train(data.train, data.dev)


    #data = read_dataset_splits(reader=data_readers.read_question_and_context_data, window_size=5, include_question_text=True, include_context_text=True, include_context_speaker=False, include_context_times=False)
    #for window_size in [1,3,5]:
    #    texts = ["turn_text-%d" % i for i in range(1, window_size+1)]
    #    model = models.MultiTextSVM(texts)
    #    trainer = SklearnTrainer(model, data_name="question_and_context_text_%d" % window_size, n_samples=5)
    #    trainer.train(data.train, data.dev)
    
    # Question text + Question duration (D)

    data = read_dataset_splits(reader=data_readers.read_question_and_duration_data)
    
    print("SVM Q+D")
    trainer = SklearnTrainer(models.SVMWithScalar("question_duration_sec"), data_name="question_and_duration", n_samples=5)
    trainer.train(data.train, data.dev)
    
    print("LR Q+D")    
    trainer = SklearnTrainer(models.LogisticWithScalar("question_duration_sec"), data_name="question_and_duration", n_samples=5)
    trainer.train(data.train, data.dev)
    
    #data = read_dataset_splits(reader=data_readers.read_question_and_index_data)
    #trainer = SklearnTrainer(models.LogisticWithScalar("question_index"), data_name="question_and_index", n_samples=5)
    #trainer.train(data.train, data.dev)
    #trainer = SklearnTrainer(models.SVMWithScalar("question_index"), data_name="question_and_index", n_samples=5)
    #trainer.train(data.train, data.dev)
    

    #data = read_dataset_splits(reader=data_readers.read_question_only_data)
    #trainer = SklearnTrainer(models.Logistic, data_name="question_only", n_samples=5)
    #trainer.train(data.train, data.dev)
    #trainer = SklearnTrainer(models.SVM, data_name="question_only", n_samples=5)
    #trainer.train(data.train, data.dev)
    #trainer = SklearnTrainer(models.Dummy, data_name="question_only", n_samples=1)
    #trainer.train(data.train, data.dev)
    
    print('All models done!\n')

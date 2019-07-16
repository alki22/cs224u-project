import argparse
import data_readers
import data_util
import numpy as np
import os
import pandas as pd
import progressbar

from collections import defaultdict
from config import Config
from console import log_info, log_warning
from enum import Enum
from model_utils import get_response_time_label, add_question_length
from multiprocessing import Pool
from pathlib import Path
from stanfordcorenlp import StanfordCoreNLP

COUNTRY_TO_INDEX = {'NO_COUNTRY': 0, 'HUN': 1, 'USA': 2, 'US': 2, 'PHL': 3, 'KOR': 4, 'BIH': 5, 'CA': 6, 'SGP': 7, 'AR': 8, 'IN': 9, 'GBR': 10, 'SG': 11, 'VN': 12, 'JAM': 13, 'PH': 14, 'FRA': 15, 'PK': 16}

class Dataset(Enum):
    QUESTION_ONLY = 1
    QUESTION_AND_CONTEXT_WINDOW = 2
    QUESTION_TEXT_AND_RESPONSE_TEXT = 5
    QUESTION_AND_INDEX = 6
    QUESTION_AND_DURATION= 7
    QUESTION_AND_NEWLINES = 8
    QUESTION_AND_SENTIMENT = 9
    QUESTION_AND_TLDX = 10
    QUESTION_AND_PAYMENT_INFO = 11
    QUESTION_AND_TUTOR_COUNTRY = 12
    QUESTION_AND_TUTOR_SCORE = 13
    LABEL_COUNTS = 14

def build_question_only(split="tiny", concatenator=None):
    data = data_readers.read_corpus(split)
    questions = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    progress = progressbar.ProgressBar(max_value=len(sessions)).start()
    for i, session in enumerate(sessions):
        for question, response in session.iter_question_and_response(concatenator=concatenator):
            questions.append(question.row.text)
            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)
        progress.update(i)

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "response_time_sec": response_times_sec})
    progress.finish()
    return dataset

def build_question_and_index(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    question_indices = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        for (question_index, (question, response)) in enumerate(session.iter_question_and_response()):
            questions.append(question.row.text)
            question_indices.append(question_index)
            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "question_index": question_indices, "response_time_sec": response_times_sec})
    return dataset

def build_question_and_duration(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    question_durations_sec = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        for question, response in session.iter_question_and_response():
            questions.append(question.row.text)
            question_durations_sec.append(question.duration)
            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "question_duration_sec": question_durations_sec, "response_time_sec": response_times_sec})
    return dataset

NLP = StanfordCoreNLP(Config.CORE_NLP_DIR)
def get_mean_sentiment(text):
    try:
        annotated = NLP._request(annotators="sentiment", data=text)
    except:
        log_warning("Sentiment annotation failed. Defaulting to neutral sentiment")
        print("\t" + text)
        return 2
    return np.mean([int(sentence["sentimentValue"]) for sentence in annotated["sentences"]])

def process_session(session):
    res = defaultdict(list)
    for question, response in session.iter_question_and_response():
        res["questions"].append(question.row.text)
        res["response_times_sec"].append((response.row.created_at - question.row.created_at).seconds)
        res["session_ids"].append(session.id)
        sentiment = get_mean_sentiment(" ".join(question.row.text))
        res["sentiments"].append(sentiment)
    return res

def build_question_and_sentiment(split="tiny"):
    data = data_readers.read_corpus(split)

    pool = Pool(12)
    sessions = data_util.get_sessions(data)

    combined_results = defaultdict(list)
    for session_result in progressbar.progressbar(pool.imap(process_session, sessions), max_value=len(sessions)):
        for k, v in session_result.items():
            combined_results[k].extend(v)
    session_ids = combined_results["session_ids"]
    questions = combined_results["questions"]
    sentiments = combined_results["sentiments"]
    response_times_sec = combined_results["response_times_sec"]

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "question_sentiment": sentiments, "response_time_sec": response_times_sec})
    return dataset

def build_question_with_context_window(split="tiny", window_size=0):
    data = data_readers.read_corpus(split)
    sessions = data_util.get_sessions(data)

    questions = []
    response_times_sec = []
    session_ids = []
    turn_texts = defaultdict(list)
    turn_speakers = defaultdict(list)
    turn_times = defaultdict(list)

    for session in progressbar.progressbar(sessions):
        for question, response in session.iter_question_and_response():
            questions.append(question.row.text)
            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)

            times = defaultdict(lambda: 0)
            texts = defaultdict(lambda: [])
            speakers = defaultdict(lambda: Config.EMPTY_TAG)
            prev = question.row.created_at
            for i, turn in enumerate(session.iter_turns(start_row=question.index, num_turns=window_size+1, direction=-1)):
                texts[i] = turn.text
                speakers[i] = turn.sent_from
                times[i] = int((prev - turn.created_at).seconds)
                prev = turn.created_at

            for i in range(1, window_size+1):
                turn_texts["turn_text-%d" % i].append(texts[i])
                turn_speakers["turn_speaker-%d" % i].append(speakers[i])
                turn_times["turn_time-%d" % i].append(times[i])

    columns = {"session_id": session_ids, "question": questions, "response_time_sec": response_times_sec}
    columns.update(turn_texts)
    columns.update(turn_speakers)
    columns.update(turn_times)
    dataset = pd.DataFrame.from_dict(columns)
    return dataset

def build_question_and_tldx(split="tiny", window_size=5):
    data = data_readers.read_corpus(split)
    sessions = data_util.get_sessions(data)

    questions = []
    response_times_sec = []
    session_ids = []
    question_durations_sec = []
    question_lengths = []
    turn_texts = defaultdict(list)
    turn_speakers = defaultdict(list)
    turn_times = defaultdict(list)

    for session in progressbar.progressbar(sessions):
        for question, response in session.iter_question_and_response():
            questions.append(question.row.text)
            question_lengths.append(len(question.row.text))
            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            question_durations_sec.append(question.duration)
            session_ids.append(session.id)

            times = defaultdict(lambda: 0)
            texts = defaultdict(lambda: [])
            speakers = defaultdict(lambda: Config.EMPTY_TAG)
            prev = question.row.created_at
            for i, turn in enumerate(session.iter_turns(start_row=question.index, num_turns=window_size+1, direction=-1)):
                texts[i] = turn.text
                speakers[i] = turn.sent_from
                times[i] = int((prev - turn.created_at).seconds)
                prev = turn.created_at

            for i in range(1, window_size+1):
                turn_texts["turn_text-%d" % i].append(texts[i])
                turn_speakers["turn_speaker-%d" % i].append(speakers[i])
                turn_times["turn_time-%d" % i].append(times[i])

    columns = {"session_id": session_ids, "question": questions, "question_length": question_lengths, 
                    "question_duration_sec": question_durations_sec, "response_time_sec": response_times_sec}
    columns.update(turn_texts)
    columns.update(turn_speakers)
    columns.update(turn_times)
    dataset = pd.DataFrame.from_dict(columns)
    return dataset

def build_question_and_payment_info(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    response_times_sec = []
    payment_infos = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        for question, response in session.iter_question_and_response():
            session_payment_info = None
            questions.append(question.row.text)
            if (question.row.student_has_payment_info == True or question.row.student_has_payment_info == False):
                payment_infos.append(question.row.student_has_payment_info)
                session_payment_info = question.row.student_has_payment_info
            elif (session_payment_info == None):
                payment_infos.append(False)
            else:
                tutor_scores.append(session_payment_info)

            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "has_payment_info": payment_infos, "response_time_sec": response_times_sec})
    return dataset

def build_question_and_tutor_country(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    response_times_sec = []
    tutor_countries = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        for question, response in session.iter_question_and_response():
            session_tutor_country = COUNTRY_TO_INDEX["NO_COUNTRY"]
            questions.append(question.row.text)
            if (type(question.row.tutor_last_sign_in_country) != str or not question.row.tutor_last_sign_in_country.isalpha() or question.row.tutor_last_sign_in_country in ["nan", "--"]):
                tutor_countries.append(session_tutor_country) 
            else:
                tutor_countries.append(COUNTRY_TO_INDEX[question.row.tutor_last_sign_in_country])
                session_tutor_country = COUNTRY_TO_INDEX[question.row.tutor_last_sign_in_country]

            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)
    print(set(tutor_countries))
    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "tutor_country": tutor_countries, "response_time_sec": response_times_sec})
    return dataset

def build_question_and_tutor_score(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    tutor_scores = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        session_tutor_score = -1
        for question, response in session.iter_question_and_response():
            questions.append(question.row.text)
            if (question.row.subject == "Math" and type(question.row.tutor_math_exam_score) is int):
                tutor_scores.append(int(question.row.tutor_math_exam_score))
                session_tutor_score = int(question.row.tutor_math_exam_score)
            elif (question.row.subject == "Physics" and type(question.row.tutor_physics_exam_score) is int):
                tutor_scores.append(int(question.row.tutor_physics_exam_score))
                session_tutor_score = int(question.row.tutor_physics_exam_score)
            elif (question.row.subject == "Chemistry" and type(question.row.tutor_chemistry_exam_score) is int):
                tutor_scores.append(int(question.row.tutor_chemistry_exam_score))
                session_tutor_score = int(question.row.tutor_chemistry_exam_score)
            else:
                tutor_scores.append(int(session_tutor_score))

            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)

    print(len(session_ids), len(questions), len(tutor_scores), len(response_times_sec))
    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "tutor_score": tutor_scores, "response_time_sec": response_times_sec})
    return dataset

def build_question_text_and_response_text(split="tiny"):
    data = data_readers.read_corpus(split)
    questions = []
    responses = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        for question, response in session.iter_question_and_response():
            questions.append(question.row.text)
            responses.append(response.row.text)
            response_times_sec.append((response.row.created_at - question.row.created_at).seconds)
            session_ids.append(session.id)

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "response": responses, "response_time_sec": response_times_sec})
    return dataset

def build_label_counts(split="tiny"):
    data = data_readers.read_corpus(split)
    label_counts = []
    response_times_sec = []
    session_ids = []

    sessions = data_util.get_sessions(data)
    for session in progressbar.progressbar(sessions):
        counts = defaultdict(int)
        for question, response in session.iter_question_and_response():
            response_time_sec = (response.row.created_at - question.row.created_at).seconds
            response_times_sec.append(response_time_sec)
            label_counts.append(tuple(counts[label] for label in Config.LABELS))
            session_ids.append(session.id)

            counts[get_response_time_label(response_time_sec)] += 1

    dataset = pd.DataFrame.from_dict({"session_id": session_ids,  "response_time_sec": response_times_sec, "label_counts": label_counts})
    return dataset

def get_dest_name(split="tiny"):
    destname = "%s_%s_dataset.csv" % (split, args.dataset.name.lower())
    dest = os.path.join(Config.DATA_DIR, destname)
    return dest

if __name__ == "__main__":
    assert Path(Config.CORPUS_FILE).exists(), "%s does not exist" % Config.CORPUS_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", type=str, default=Dataset.QUESTION_ONLY.name,
            help="Which dataset to build. Defaults to QUESTION_ONLY")
    args = parser.parse_args()
    args.dataset = Dataset[args.dataset]

    builders = {Dataset.QUESTION_ONLY: build_question_only,
                Dataset.QUESTION_AND_INDEX: build_question_and_index,
                Dataset.QUESTION_AND_DURATION: build_question_and_duration,
                Dataset.QUESTION_AND_SENTIMENT: build_question_and_sentiment,
                Dataset.QUESTION_AND_NEWLINES: lambda split: build_question_only(split, concatenator="\n"),
                Dataset.QUESTION_AND_CONTEXT_WINDOW: lambda split: build_question_with_context_window(split, window_size=Config.MAX_CONTEXT_WINDOW_SIZE),
                Dataset.QUESTION_TEXT_AND_RESPONSE_TEXT: build_question_text_and_response_text,
                Dataset.QUESTION_AND_TLDX: build_question_and_tldx,
                Dataset.QUESTION_AND_PAYMENT_INFO: build_question_and_payment_info,
                Dataset.QUESTION_AND_TUTOR_COUNTRY: build_question_and_tutor_country,
                Dataset.QUESTION_AND_TUTOR_SCORE: build_question_and_tutor_score,
                Dataset.LABEL_COUNTS: build_label_counts}

    log_info("Building the %s dataset" % args.dataset.name.lower())

    for split in Config.SPLITS:
        log_info("Building %s" % split)
        dataset = builders[args.dataset](split)
        print("\tExtracted %s samples" % dataset.shape[0])

        dest = get_dest_name(split)
        print("\tWriting dataset to %s" % dest)
        dataset.to_csv(dest, index=False)

from os.path import dirname, realpath, join
from pathlib import Path

class Config:
    BASE_DIR = Path(realpath(join(dirname(realpath(__file__)), '..')))
    RUNS_DIR = Path(join(BASE_DIR, "runs"))
    DATA_DIR = Path(join(BASE_DIR, "data"))
    MODELS_DIR = Path(join(BASE_DIR, "models"))
    CORPUS_FILE = Path(join(DATA_DIR, "new_dataset_preprocessed.csv"))
    FASTTEXT_FILE = Path(join(DATA_DIR, "fasttext", "wiki.en.vec")) 
    CORE_NLP_DIR = join(BASE_DIR, "stanford-corenlp-full-2018-10-05")
    BASELINE_PREDS_FILE = Path(join(DATA_DIR, "dev_baseline_predictions_logreg.csv"))
    SPLITS = ["tiny", "train", "dev", "test"]

    def _lift_split_file(fmt):
        def split_file(split):
            assert split in Config.SPLITS
            return Path(join(Config.DATA_DIR, fmt % split))
        return split_file

    CORPUS_SPLIT_FILE = _lift_split_file("%s_new_dataset_preprocessed.csv") #default=CORPUS_SPLIT_FILE = _lift_split_file("%s_yup_messages_preprocessed.csv")
    QUESTION_ONLY_DATASET_FILE = _lift_split_file("%s_question_only_dataset.csv")
    QUESTION_AND_INDEX_DATASET_FILE = _lift_split_file("%s_question_and_index_dataset.csv")
    QUESTION_AND_DURATION_DATASET_FILE = _lift_split_file("%s_question_and_duration_dataset.csv")
    QUESTION_AND_NEWLINES_DATASET_FILE = _lift_split_file("%s_question_and_newlines_dataset.csv")
    QUESTION_AND_TLDX_DATASET_FILE = _lift_split_file("%s_question_and_tldx_dataset.csv")
    QUESTION_AND_PAYMENT_INFO_DATASET_FILE = _lift_split_file("%s_question_and_payment_info_dataset.csv")
    QUESTION_AND_TUTOR_COUNTRY_DATASET_FILE = _lift_split_file("%s_question_and_tutor_country_dataset.csv")
    QUESTION_AND_TUTOR_SCORE_DATASET_FILE = _lift_split_file("%s_question_and_tutor_score_dataset.csv")
    QUESTION_AND_CR_PREDICTION_DATASET_FILE = _lift_split_file("%s_question_and_cr_prediction_dataset.csv")
    QUESTION_TEXT_AND_RESPONSE_TEXT_DATASET_FILE = _lift_split_file("%s_question_text_and_response_text_dataset.csv")
    QUESTION_AND_SENTIMENT_DATASET_FILE = _lift_split_file("%s_question_and_sentiment_dataset.csv")
    LABEL_COUNTS_DATASET_FILE = _lift_split_file("%s_label_counts_dataset.csv")
    CR_CLASSIFIER_DATASET_FILE = Path(join(DATA_DIR, "question_text_and_response_text_dataset.csv"))

    LR_CR_CLASSIFIER_MODEL = Path(join(MODELS_DIR, "cr_classifier", "logistic", "clf.sav"))
    SVM_CR_CLASSIFIER_MODEL = Path(join(MODELS_DIR, "cr_classifier", "svm", "clf.sav"))

    MAX_CONTEXT_WINDOW_SIZE = 10
    QUESTION_AND_CONTEXT_WINDOW_DATASET_FILE = _lift_split_file("%s_question_and_context_window_dataset.csv")

    EMPTY_TAG = "<empty>"
    URL_TAG = "<url>"
    REMOVED_ROWS_FILE = Path(join(DATA_DIR, "removed_rows.csv"))

    STUDENT_SPEAKERS = ["student"]
    TUTOR_SPEAKERS = ["tutor"]
    SYSTEM_SPEAKERS =  ["system info", "system alert", "system warn", "bot"]
    PLATFORM_SPEAKERS = TUTOR_SPEAKERS  + SYSTEM_SPEAKERS

    LABEL_SHORT = "short"
    LABEL_MEDIUM = "medium"
    LABEL_LONG = "long"
    #LABELS = [LABEL_SHORT, LABEL_MEDIUM, LABEL_LONG]
    LABELS = [LABEL_SHORT, LABEL_LONG]
    #THRESHOLD_SHORT = 15
    #THRESHOLD_MEDIUM = 45
    THRESHOLD = 21

    SEED = 42 #For reproducibility





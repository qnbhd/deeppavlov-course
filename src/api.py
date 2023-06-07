from pathlib import Path

import joblib
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import NearestCentroid

nlp = spacy.load('en_core_web_sm')


def preprocess_spacy(
    dataframe: pd.DataFrame,
    remove_punct: bool,
    remove_like_num: bool,
    remove_like_email: bool,
    remove_stop: bool,
):
    """
    Preprocesses the text in the given dataframe using Spacy.

    Args:
        dataframe (pandas.DataFrame): The input dataframe containing the 'comment' column.
        remove_punct (bool): Whether to remove punctuation.
        remove_like_num (bool): Whether to remove tokens that resemble numbers.
        remove_like_email (bool): Whether to remove tokens that resemble email addresses.
        remove_stop (bool): Whether to remove stop words.

    Returns:
        The preprocessed dataframe with the 'comment' column modified.

    """
    def callee(t):
        result = True
        result = (result and not t.is_punct) if remove_punct else result
        result = (result and not t.like_num) if remove_like_num else result
        result = (result and not t.like_email) if remove_like_email else result
        result = (result and not t.is_stop) if remove_stop else result
        return result

    dataframe = dataframe.copy()

    dataframe['comment'] = dataframe['comment'].apply(
        lambda x: ' '.join([
            tok.lemma_.lower() for tok in nlp(x) if callee(tok)
        ])
    )

    return dataframe


def train_model(train_dataset: pd.DataFrame, embeddings_model: str, metric: str):
    """
    Trains a classification model using SentenceTransformer embeddings and NearestCentroid classifier.

    Args:
        train_dataset (pandas.DataFrame): The training dataset containing the 'comment' and 'isHate' columns.
        embeddings_model (str): The name of the SentenceTransformer model to use for generating embeddings.
        metric (str): The distance metric to use for the NearestCentroid classifier.

    Returns:
        sklearn.neighbors.NearestCentroid: The trained NearestCentroid classifier.

    """
    model = SentenceTransformer(embeddings_model)
    embeddings = model.encode(train_dataset['comment'].values, show_progress_bar=True)
    clf = NearestCentroid(metric=metric)
    clf.fit(embeddings, train_dataset["isHate"])
    return clf


def evaluate_model(test_dataset: pd.DataFrame, model_path: Path, embeddings_model: str):
    """
    Evaluates a classification model on a test dataset using SentenceTransformer embeddings and pre-trained classifier.

    Args:
        test_dataset (pandas.DataFrame): The test dataset containing the 'comment' and 'isHate' columns.
        model_path (str): The path to the pre-trained NearestCentroid classifier.
        embeddings_model (str): The name of the SentenceTransformer model to use for generating embeddings.

    Returns:
        dict: A dictionary containing evaluation metrics including F1 score, ROC AUC score,
              precision, recall, and accuracy.

    """
    model = SentenceTransformer(embeddings_model)
    clf = joblib.load(model_path)
    embeddings = model.encode(test_dataset["comment"].values)
    predictions = clf.predict(embeddings)
    metrics = {
        'F1': f1_score(test_dataset['isHate'], predictions),
        'ROC AUC': roc_auc_score(test_dataset['isHate'], predictions),
        'Precision': precision_score(test_dataset['isHate'], predictions),
        'Recall': recall_score(test_dataset['isHate'], predictions),
        'Accuracy': accuracy_score(test_dataset['isHate'], predictions),
    }
    return metrics

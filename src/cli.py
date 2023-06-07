"""This module provides CLI interface."""
import json
from pathlib import Path

import click
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from nltk import (
    NLTKWordTokenizer,
    TweetTokenizer,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, \
    accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline

import spacy

nlp = spacy.load('en_core_web_sm')


def string_to_bool(ctx, param, value):
    if value.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise click.BadParameter('Invalid value for --foo. Must be true/false.')


def validate_input_dataframe(ctx, param, value):
    try:
        df = pd.read_csv(value, sep=';')
    except pd.errors.ParserError:
        raise click.BadParameter("Input dataset must be in `csv` format.")

    if "comment" not in df.columns or "isHate" not in df.columns:
        raise click.BadParameter(
            "Input dataframe must have `comment` and `isHate` columns."
        )

    return df


def preprocess_spacy(
    dataframe,
    remove_punct,
    remove_like_num,
    remove_like_email,
    remove_stop
):
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


@click.group()
def cli():
    pass


@click.command()
@click.argument(
    "dataset",
    type=click.Path(exists=True, path_type=Path),
    callback=validate_input_dataframe,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output to preprocessed data",
    default=Path.cwd().joinpath("preprocessed"),
)
@click.option('--remove-punct', callback=string_to_bool, default='true')
@click.option('--remove-emails', callback=string_to_bool, default='true')
@click.option('--remove-numbers', callback=string_to_bool, default='false')
@click.option('--remove-stop', callback=string_to_bool, default='true')
def preprocess(dataset, output, remove_punct, remove_emails, remove_numbers, remove_stop):
    dataset_preproc = preprocess_spacy(dataset, remove_punct, remove_numbers, remove_emails,  remove_stop)
    dataset_preproc['isHate'] = dataset['isHate'].astype(int)
    dataset_preproc.to_csv(output, index=None, sep=';')


@click.command()
@click.argument(
    "dataset",
    type=click.Path(exists=True, path_type=Path),
    callback=validate_input_dataframe,
)
@click.option("--test-size", type=float, default=0.2)
@click.option("--random-state", type=int, default=42)
@click.option(
    "--output-train",
    type=click.Path(path_type=Path),
    help="Output to train data",
    default=Path.cwd().joinpath("train.csv"),
)
@click.option(
    "--output-test",
    type=click.Path(path_type=Path),
    help="Output to test data",
    default=Path.cwd().joinpath("test.csv"),
)
def split(dataset, test_size, random_state, output_train, output_test):
    train, test = train_test_split(
        dataset, test_size=test_size, random_state=random_state, stratify=dataset['isHate']
    )
    train.to_csv(output_train, index=None, sep=';')
    test.to_csv(output_test, index=None, sep=';')


@click.command()
@click.argument(
    "train-dataset",
    type=click.Path(exists=True, path_type=Path),
    callback=validate_input_dataframe,
)
@click.option("--embeddings-model", type=str)
@click.option("--metric", type=click.Choice(["euclidean", "manhattan"]))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output to trained model",
    default=Path.cwd().joinpath("model.joblib"),
)
def train(train_dataset, embeddings_model, metric, output):
    model = SentenceTransformer(embeddings_model)
    embeddings = model.encode(train_dataset['comment'].values, show_progress_bar=True)
    clf = NearestCentroid(metric=metric)
    clf.fit(embeddings, train_dataset["isHate"])
    joblib.dump(clf, output)


@click.command()
@click.argument(
    "test-dataset",
    type=click.Path(exists=True, path_type=Path),
    callback=validate_input_dataframe,
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Joblib dump of model",
)
@click.option("--embeddings-model", type=str)
@click.option(
    "--report-output",
    type=click.Path(path_type=Path),
    help="Report to output",
)
def evaluate(test_dataset, model_path, embeddings_model, report_output):
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
    Path(report_output.parent).mkdir(parents=True, exist_ok=True)
    with open(report_output, "w") as f:
        f.write(json.dumps(metrics, indent=4))


cli.add_command(preprocess)
cli.add_command(split)
cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

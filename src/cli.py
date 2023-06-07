"""This module provides CLI interface."""
import json
from pathlib import Path

import click
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.api import preprocess_spacy, evaluate_model, train_model


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
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=random_state, stratify=dataset['isHate']
    )
    train_dataset.to_csv(output_train, index=None, sep=';')
    test_dataset.to_csv(output_test, index=None, sep=';')


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
    clf = train_model(train_dataset, embeddings_model, metric)
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
    metrics = evaluate_model(test_dataset, model_path, embeddings_model)
    Path(report_output.parent).mkdir(parents=True, exist_ok=True)
    with open(report_output, "w") as f:
        f.write(json.dumps(metrics, indent=4))


cli.add_command(preprocess)
cli.add_command(split)
cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

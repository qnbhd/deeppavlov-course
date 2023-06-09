import json
import subprocess
from pathlib import Path

import pandas as pd
import spacy
import optuna
from dvclive.optuna import DVCLiveCallback


data_folder = Path(__file__).parent.parent.joinpath("data")
train = pd.read_csv(data_folder.joinpath("train_raw.csv"))
test = pd.read_csv(data_folder.joinpath("test_raw.csv"))
train['isHate'] = train['isHate'].astype(int)
test['isHate'] = test['isHate'].astype(int)

nlp = spacy.load("en_core_web_sm")


def objective(trial):
    # preprocessing
    remove_punct = trial.suggest_categorical('preprocess.remove_punct', [True, False])
    remove_like_num = trial.suggest_categorical('preprocess.remove_number', [True, False])
    remove_like_email = trial.suggest_categorical('preprocess.remove_emails', [True, False])
    remove_stop = trial.suggest_categorical('preprocess.remove_stop', [True, False])

    # sentence-transformers
    model_kind = trial.suggest_categorical('train.embeddings_model', [
        'all-mpnet-base-v2',
        'multi-qa-mpnet-base-dot-v1',
        'all-distilroberta-v1',
        'all-MiniLM-L12-v2',
        'multi-qa-distilbert-cos-v1',
        'all-MiniLM-L6-v2',
        'paraphrase-albert-small-v2',
    ])

    metric = trial.suggest_categorical('train.metric', ['manhattan', 'euclidean'])

    subprocess.run(
        [
            "dvc", "exp", "run",
            "--set-param", f"preprocess.remove_punct={remove_punct}",
            "--set-param", f"preprocess.remove_emails={remove_like_email}",
            "--set-param", f"preprocess.remove_numbers={remove_like_num}",
            "--set-param", f"preprocess.remove_stop={remove_stop}",
            "--set-param", f"train.embeddings_model={model_kind}",
            "--set-param", f"train.metric={metric}",
        ]
    )

    with open(Path(__file__).parent.parent.joinpath("eval", "live", "metrics.json"), 'r') as f:
        metrics = json.loads(f.read())

    return metrics['ROC AUC']


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective, n_trials=20, callbacks=[DVCLiveCallback()], show_progress_bar=True)

import hashlib
import sys
from pathlib import Path
from typing import Optional

import click
import requests
from rich.progress import Progress

DOWNLOAD_LINK = 'https://raw.githubusercontent.com/intelligence-csd-auth-gr/' \
                'Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv'
MD5_HASHSUM = 'd475a0a4e09db8d6a405b6c9fb962da3'
FILENAME = Path(DOWNLOAD_LINK).name


def hashsum(path):
    """
    Calculate md5 hash of a file

    Parameters
    ----------
    path : PathLike
        Path to the file

    Returns
    -------
    str
        md5 hash of the file
    """

    hash_inst = hashlib.md5()  # nosec
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(hash_inst.block_size * 128), b""):
            hash_inst.update(chunk)
    return hash_inst.hexdigest()


@click.command()
def download():
    data_folder = Path(__file__).parent.parent.joinpath('data')
    dataset_path = data_folder.joinpath(FILENAME)

    with open(dataset_path, "wb") as f:
        response = requests.get(DOWNLOAD_LINK, stream=True)
        total_length_header: Optional[str] = response.headers.get(
            "Content-Length"
        )

        if total_length_header is None:
            f.write(response.content)
        else:
            total_length = int(total_length_header)
            with Progress() as progress:
                task = progress.add_task(
                    f"[red]Downloading `{FILENAME}`...", total=total_length
                )
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    progress.update(task, advance=len(data))

        # verify md5 hash
    if hashsum(dataset_path) != MD5_HASHSUM:
        raise ValueError(f"Downloaded dataset file `{FILENAME}` has wrong md5 hash")

        # in jupyter notebook
    if "ipykernel" in sys.modules:
        from IPython.display import clear_output  # noqa

        clear_output(wait=False)


if __name__ == '__main__':
    download()

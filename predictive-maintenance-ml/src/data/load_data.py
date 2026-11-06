from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib import request

import pandas as pd

from src.utils.config import CONFIG, PROJECT_ROOT

DATA_RAW_DIR = PROJECT_ROOT / CONFIG['paths']['data_raw']

DEFAULT_COLUMN_NAMES = (
    ['engine_id', 'cycle']
    + [f'setting_{idx}' for idx in range(1, 4)]
    + [f'sensor_{idx}' for idx in range(1, 22)]
)


def ensure_data_directory(path: str | Path = DATA_RAW_DIR) -> Path:
    """Create the raw data directory if it does not exist."""
    data_dir = Path(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def generate_dataset_urls(dataset_name: str) -> list[str]:
    """Build a small set of candidate dataset URLs."""
    base_urls = CONFIG['cmapss']['source_urls']
    file_names = [
        f'train_{dataset_name}.txt',
        f'test_{dataset_name}.txt',
        f'RUL_{dataset_name}.txt',
    ]
    urls: list[str] = []
    for base in base_urls:
        for file_name in file_names:
            candidate = f'{base.rstrip("/")}/{file_name}'
            urls.append(candidate)
    return urls


def download_file_if_missing(url: str, destination: str | Path) -> Path:
    """Download a single file from a URL if it is not already present."""
    destination_path = Path(destination)
    if destination_path.exists():
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(url) as response, open(destination_path, 'wb') as file_handle:
        file_handle.write(response.read())
    return destination_path


def acquire_cmapss_dataset(dataset_name: str, raw_dir: str | Path = DATA_RAW_DIR) -> list[Path]:
    """Download the complete FD dataset family for a given dataset subset if needed.

    The function keeps the original raw files in place and returns the local paths.
    """
    raw_data_dir = ensure_data_directory(raw_dir)
    dataset_files: list[Path] = []
    for file_suffix in ['train', 'test', 'RUL']:
        file_name = f'{file_suffix}_{dataset_name}.txt'
        file_path = raw_data_dir / file_name
        if file_path.exists():
            dataset_files.append(file_path)
            continue

        candidates = [
            f'https://zenodo.org/records/15346912/files/{file_name}?download=1',
            f'https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/{file_name}',
            f'https://raw.githubusercontent.com/jean-paulb/NASA-CMAPSS-dataset/master/{file_name}',
        ]
        downloaded = False
        for candidate in candidates:
            try:
                download_file_if_missing(candidate, file_path)
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            raise FileNotFoundError(
                f"Unable to download {file_name}. Please download manually into {raw_data_dir}."
            )
        dataset_files.append(file_path)
    return dataset_files


def load_cmapss_dataset(file_path: str | Path, dataset_label: str | None = None) -> pd.DataFrame:
    """Load a C-MAPSS train or test file into a DataFrame with standard names."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist.')

    data = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    if data.shape[1] != len(DEFAULT_COLUMN_NAMES):
        raise ValueError(
            f'{path.name} has {data.shape[1]} columns; expected {len(DEFAULT_COLUMN_NAMES)}. '
            'Check the raw dataset format before proceeding.'
        )

    data.columns = DEFAULT_COLUMN_NAMES
    if dataset_label is not None:
        data['dataset'] = dataset_label
    return data


def collect_fd_dataset(fd_id: str = 'FD001', raw_dir: str | Path = DATA_RAW_DIR) -> dict[str, pd.DataFrame]:
    """Return a dictionary containing the train, test and RUL DataFrames for a given FD subset."""
    files = acquire_cmapss_dataset(fd_id, raw_dir=raw_dir)
    file_map = {
        'train': next(path for path in files if path.name.startswith('train_')),
        'test': next(path for path in files if path.name.startswith('test_')),
        'RUL': next(path for path in files if path.name.startswith('RUL_')),
    }
    return {
        'train': load_cmapss_dataset(file_map['train'], dataset_label=fd_id),
        'test': load_cmapss_dataset(file_map['test'], dataset_label=fd_id),
        'rul': pd.read_csv(file_map['RUL'], sep=r'\s+', header=None, engine='python', names=['rul'])
    }


def list_dataset_files(raw_dir: str | Path = DATA_RAW_DIR) -> list[Path]:
    """Return the dataset files presently available in the raw data directory."""
    return sorted(Path(raw_dir).glob('*'))

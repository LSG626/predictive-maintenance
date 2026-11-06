# Raw dataset storage

Place the original NASA C-MAPSS files in this directory without modifying them.

Expected files for the full research workflow include:
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt
- train_FD002.txt
- test_FD002.txt
- RUL_FD002.txt
- train_FD003.txt
- test_FD003.txt
- RUL_FD003.txt
- train_FD004.txt
- test_FD004.txt
- RUL_FD004.txt

## Manual download instructions
1. Visit the official NASA repository or the Zenodo dataset record.
2. Download the C-MAPSS turbofan engine degradation data package.
3. Extract the files into this directory.
4. Keep the original raw files unchanged for provenance and reproducibility.

## Validation requirement
The project code in `src/data/load_data.py` is designed to validate structure, engine counts, and missing values before any feature engineering begins.

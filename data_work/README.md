# Data Work

This folder is intended for researchers who want to generate data, train models, or otherwise
run data preparation scripts outside of the core LatentScore package.

## Environment setup

Create a local Python virtual environment and install dependencies:

```bash
./data_work/setup_python_env.sh
source data_work/.venv/bin/activate
```

You can also do it manually:

```bash
python3 -m venv data_work/.venv
source data_work/.venv/bin/activate
pip install -r data_work/requirements.txt
```

## Scripts

### `0_download_base_data.py`

Downloads Common Pile datasets, samples 1,000 texts from each, and writes JSONL files with
standardized fields (`created`, `metadata`, `dataset`, `id_in_dataset`, `text`). The script
prints approximate download sizes and prompts for confirmation before downloading.

Usage example:

```bash
<<<<<<< ours
python data_work/0_download_base_data.py --seed 123 --output-dir data_work/.outputs
=======
python data_work/0_download_base_data.py --seed 123 --sample-size 1000 --output-dir data_work/.outputs
>>>>>>> theirs
```

### `1_process_base_data.py`

Validates that downloaded JSONL files match the expected schema, then (eventually) will
split them into train/test/eval data. Currently, it stops with a `NotImplementedError` after
schema validation.

Usage example:

```bash
python data_work/1_process_base_data.py --input-dir data_work/.outputs
```

Use `-h` on each script for detailed CLI help and arguments.

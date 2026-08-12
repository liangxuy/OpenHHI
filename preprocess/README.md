# Inter-X Data Preprocessing

This directory provides the preprocessing pipeline for preparing Inter-X motion and text data. The motion pipeline converts paired SMPL-X parameters to continuous 6D rotations (Rot6D), while the text pipeline tokenizes captions and builds a dataset-specific vocabulary from pretrained GloVe embeddings.

## Overview

### Motion Pipeline

```text
Inter-X motions/<sequence>/{P1,P2}.npz
                 │
                 ▼
1_prepare_data_npy.py
                 │  [T, 56, 6] float32
                 ▼
interx_smplx_npy/<sequence>.npy
                 │
                 ▼
2_rot6d.py
                 │  one [T, 56, 12] dataset per sequence
                 ▼
interx_smplx_rot6d.h5
                 │
                 ▼
3_split_train_test.py
                 │
                 ├── train.h5
                 ├── val.h5
                 └── test.h5
```

### Text Pipeline

```text
Inter-X texts/<sequence>.txt       glove.6B.300d.txt
                 │                         │
                 └──────────┬──────────────┘
                            ▼
                    4_text_process.py
                            │
                            ├── texts_processed/<sequence>.txt
                            └── glove/
                                ├── interx_vab_idx.pkl
                                ├── interx_vab_words.pkl
                                └── interx_vab_data.npy
```

## Requirements

Python 3.9–3.11 is recommended. From this directory, create an environment and install the Python dependencies:

```bash
conda create -n interx-preprocess python=3.11
conda activate interx-preprocess
pip install -r requirements.txt
pip install spacy
python -m spacy download en_core_web_sm
```

The following external files are also required:

- the [Inter-X dataset](https://github.com/liangxuy/Inter-X), including motions, captions, and official split files;
- `SMPLX_NEUTRAL.npz` from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/); and
- the 300-dimensional `glove.6B.300d.txt` pretrained embeddings.

These assets are governed by their respective licenses and are not included in this repository.

## Motion Preprocessing

Run the motion steps in numerical order.

### 1. Prepare Paired Axis-Angle Motions

```bash
python 1_prepare_data_npy.py \
  --motions-dir /path/to/Inter-X_Dataset/motions \
  --body-model /path/to/SMPLX_NEUTRAL.npz \
  --output-dir outputs/interx_smplx_npy \
  --downsample 4
```

For every interaction sequence, the script:

1. loads the two SMPL-X `.npz` files;
2. keeps one frame every `--downsample` frames;
3. uses the neutral SMPL-X model to place each person on the ground; and
4. expresses both root translations relative to person 1 in the first frame.

CUDA is used automatically when available; otherwise the script runs on the CPU. Use `--device cpu` to force CPU execution. Existing outputs are skipped unless `--overwrite` is supplied. For a quick smoke test, use `--limit 1`.

Each output `.npy` file is a `float32` array with shape `[T, 56, 6]`:

- rows `0:55`: 55 axis-angle pose entries consisting of one root orientation, 21 body joints, three zero-filled jaw/eye placeholders, and 30 hand joints;
- row `55`: root translation; and
- last dimension: person 1 (`3` values) followed by person 2 (`3` values).

SMPL-X gender and shape parameters are not included in the training representation.

### 2. Convert Motions to Rot6D

```bash
python 2_rot6d.py \
  --input-dir outputs/interx_smplx_npy \
  --output outputs/h5_files/rot6d/interx_smplx_rot6d.h5
```

The resulting HDF5 file contains one `float32` dataset per sequence. Each dataset has shape `[T, 56, 12]`:

- rows `0:55`: person 1 Rot6D (`6` values) followed by person 2 Rot6D (`6` values);
- row `55`: `trans1 (3), velocity1 (3), trans2 (3), velocity2 (3)`.

Velocity is the root-translation difference between consecutive frames. The final-frame velocity is zero because there is no following frame. If the destination file exists, use `--overwrite` to replace it.

### 3. Create the Official Data Splits

```bash
python 3_split_train_test.py \
  --input outputs/h5_files/rot6d/interx_smplx_rot6d.h5 \
  --splits-dir /path/to/Inter-X_Dataset/splits \
  --output-dir outputs/h5_files/rot6d/motions \
  --strict
```

This creates `train.h5`, `val.h5`, and `test.h5` using the corresponding official Inter-X split lists. The script also rejects samples assigned to more than one split. With `--strict`, it fails if any split entry is absent from the input HDF5 file; this option is recommended for a complete dataset run.

## Text Preprocessing

### 4. Tokenize Captions and Build the GloVe Vocabulary

```bash
python 4_text_process.py \
  --text-dir /path/to/Inter-X_Dataset/texts \
  --glove-file /path/to/glove.6B.300d.txt \
  --processed-dir outputs/texts_processed \
  --glove-output-dir outputs/glove
```

The script applies the Inter-X text convention to every line of every input `.txt` file. It removes hyphens, discards non-alphabetic tokens, and lemmatizes nouns and verbs except for the word `left`. The processed captions use the following HumanML3D-style record format:

```text
<original caption>#<word/POS> <word/POS> ...#0.0#0.0
```

It then scans `glove.6B.300d.txt` for the words used by the processed captions and saves:

- `interx_vab_idx.pkl`: mapping from words to vocabulary indices;
- `interx_vab_words.pkl`: vocabulary words in index order; and
- `interx_vab_data.npy`: `float32` embedding matrix with shape `[V, 300]`.

The vocabulary begins with the special tokens `sos`, `eos`, and `unk`. Known Inter-X compound words that do not have a direct GloVe entry are approximated by averaging the embeddings of their components. Other missing words are reported and use `unk` downstream.

Use `--spacy-model` to select another installed English spaCy model. Existing processed caption or vocabulary files cause the command to fail unless `--overwrite` is supplied. As with the motion scripts, `--limit 1` provides a quick smoke test.

## Default Output Locations

All four scripts use `outputs/` as their default output root. Set `INTERX_OUTPUT_ROOT` to change it:

```bash
export INTERX_OUTPUT_ROOT=/path/to/output
```

When explicit output arguments are omitted, the generated files are organized as follows:

```text
$INTERX_OUTPUT_ROOT/
├── interx_smplx_npy/
│   └── <sequence>.npy
├── h5_files/rot6d/
│   ├── interx_smplx_rot6d.h5
│   └── motions/
│       ├── train.h5
│       ├── val.h5
│       └── test.h5
├── texts_processed/
│   └── <sequence>.txt
└── glove/
    ├── interx_vab_idx.pkl
    ├── interx_vab_words.pkl
    └── interx_vab_data.npy
```

Run `python <script>.py --help` for the complete command-line interface of any script.

## Script Reference

- `1_prepare_data_npy.py`: prepares paired axis-angle SMPL-X motions.
- `2_rot6d.py`: converts axis-angle poses to Rot6D and adds root velocity.
- `3_split_train_test.py`: creates official motion splits in HDF5 format.
- `4_text_process.py`: processes captions and constructs the GloVe vocabulary.
- `utils/rotation_conversions.py`: provides the rotation conversion functions used in step 2.

## License

The code in this directory is covered by [LICENSE](LICENSE). Inter-X, SMPL-X, GloVe, `human_body_prior`, spaCy, and the retained rotation utilities remain subject to their respective upstream licenses.

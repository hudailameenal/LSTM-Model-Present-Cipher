# LSTM Model for the PRESENT Cipher

Deep-learning cryptanalysis of the **PRESENT** lightweight block cipher. The project
trains ** LSTM** networks to recover plaintext directly from ciphertext,
and measures how well the attack works as the number of encryption rounds increases
(round 1 → round 31). It is a reproduction/experimentation harness around a
known-plaintext, deep-learning attack methodology, evaluated across three data
regimes: **CTT**, **NCTT**, and **RWTT**.

> ⚠️ **Research / educational use only.** This code is for studying the resistance of
> block ciphers to machine-learning-based cryptanalysis. It is *not* a production
> attack tool — full-round PRESENT is not broken here (accuracy decays toward random
> guessing as rounds increase, which is exactly what the experiment demonstrates).

---

## Table of Contents

- [Background](#background)
- [How It Works](#how-it-works)
- [Experiment Regimes](#experiment-regimes)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Notes & Caveats](#notes--caveats)
- [References](#references)

---

## Background

**PRESENT** is an ultra-lightweight [SPN](https://en.wikipedia.org/wiki/Substitution%E2%80%93permutation_network)
block cipher (ISO/IEC 29192-2) designed for constrained hardware such as RFID tags and
sensor nodes. It uses:

- **64-bit** block size
- **80-bit** key (the PRESENT-80 variant used here)
- A 4-bit **S-box** (substitution layer)
- A 64-bit **P-box** (bit-permutation layer)
- **31 rounds** + a final key XOR

A faithful, test-vector-verified implementation lives in [`cipher.py`](cipher.py).
The core research question: **can a neural network learn to invert the cipher
(recover plaintext from ciphertext) after N rounds, and how does that ability fade as
N grows toward the full 31 rounds?**

## How It Works

1. **Data generation** — plaintext/ciphertext pairs are produced by encrypting inputs
   with the reference PRESENT implementation ([`generate.py`](generate.py)).
2. **Preprocessing** — each 64-bit block is split into 8 bytes, normalized to `[0, 1]`,
   and reshaped to the LSTM input shape `(8 timesteps, 1 feature)`
   ([`preprocessing.py`](preprocessing.py)).
3. **Round-by-round training** — for each round `r` from 1 to 31, a LSTM is trained to
   map ciphertext → plaintext. Ciphertexts are **chained**: the output of round `r`
   feeds the encryption for round `r+1`, so the attack is evaluated at every reduced-round
   depth.
4. **Evaluation** — models are scored with:
   - **Restoration accuracy** — % of bytes recovered exactly
   - **Mean byte error (MBE)** — average absolute byte error, normalized
   - **Hamming / bitwise accuracy** — fraction of correctly recovered bits

## Experiment Regimes

The project runs three separate training pipelines, each targeting a different kind of
plaintext distribution:

| Regime   | Script            | Data                     | Idea |
|----------|-------------------|--------------------------|------|
| **CTT**  | `train_ctt.py`    | `dataset1`               | *Correlated Text* — highly structured, **incremental** plaintexts (fixed 5 bytes + a counter). Best-case for the attacker. |
| **NCTT** | `train_nctt.py`   | `dataset2`–`dataset4`    | *Non/Neg-Correlated Text* — trains on the correlated set, then tests generalization on **decremental** counter datasets. |
| **RWTT** | `train_rwtt.py`   | `dataset5`–`dataset7`    | *Real-World Text* — plaintexts sampled from real **PDF files** with random per-dataset keys. Hardest, most realistic regime. |

Each pipeline has a matching evaluator: `evaluate_ctt.py`, `evaluate_nctt.py`,
`evaluate_rwtt.py`, which pretty-print the saved round-by-round results and model info.

## Repository Structure

```
LSTM-Model-Present-Cipher/
├── cipher.py             # PRESENT-80 reference implementation (S-box, P-box, key schedule)
├── generate.py           # Generates plaintext/ciphertext CSV datasets from the cipher & PDFs
├── preprocessing.py      # Hex → byte-matrix → normalized LSTM tensors; train/test split
├── model.py              # Baseline LSTM architecture (build_model)
│
├── train_ctt.py          # CTT  training pipeline (dataset1)
├── train_nctt.py         # NCTT training pipeline (dataset2–4)
├── train_rwtt.py         # RWTT training pipeline (dataset5–7, real PDF data)
│
├── evaluate_ctt.py       # Display CTT results + model summary
├── evaluate_nctt.py      # Display NCTT results
├── evaluate_rwtt.py      # Display RWTT results (per dataset)
│
├── datasets/             # Generated CSVs: dataset1.csv … dataset7.csv
├── src/data/             # Mirror of the dataset CSVs (loaded by preprocessing.load_all_data)
├── models/               # Trained Keras models (.h5): present_ctt, present_nctt, rwtt_dataset5-7
├── results/              # Round-by-round metric CSVs (ctt / nctt / rwtt)
└── pdfs/                 # Source PDFs used to build the real-world (RWTT) datasets
```

## Installation

Requires **Python 3.9+**. Install the dependencies (no `requirements.txt` is shipped, so
install directly):

```bash
pip install tensorflow numpy scikit-learn optuna
```

| Package        | Used for |
|----------------|----------|
| `tensorflow`   | Keras LSTM models |
| `numpy`        | Byte/tensor manipulation |
| `scikit-learn` | Train/test splitting |
| `optuna`       | Hyperparameter search (fallback when paper hyperparameters aren't defined for a round) |

## Usage

### 1. Verify the cipher

```bash
python cipher.py            # or: python -c "from cipher import PresentCipher; PresentCipher.test()"
```

Confirms the implementation against the official PRESENT test vector
(`0x0000…0000` → `0x5579c1387b228445`).

### 2. (Re)generate datasets

```bash
python generate.py          # writes datasets/dataset1.csv … dataset7.csv
```

> Note: training loads data from `src/data/` via `preprocessing.load_all_data()`. If you
> regenerate, copy the CSVs from `datasets/` into `src/data/` (or point the loader at the
> `datasets/` folder).

### 3. Train

```bash
python train_ctt.py         # CTT  → models/present_ctt.h5,  results/ctt_round_results.csv
python train_nctt.py        # NCTT → models/present_nctt.h5, results/nctt_round_results.csv
python train_rwtt.py        # RWTT → models/rwtt_dataset{5,6,7}_final.h5, results/rwtt_round_results.csv
```

Each script trains progressively over rounds 1–31, using fixed hyperparameters where the
paper specifies them and Optuna search otherwise.

### 4. Evaluate

```bash
python evaluate_ctt.py
python evaluate_nctt.py
python evaluate_rwtt.py
```

## Datasets

All datasets are CSVs with two hex columns: `plaintext,ciphertext` (16 hex digits = 64
bits each).

| Dataset | Size (approx.) | Generation |
|---------|----------------|------------|
| `dataset1` | 2¹⁵ pairs | Incremental counter, fixed 5-byte prefix (CTT) — static test-vector key |
| `dataset2`–`dataset4` | 2¹¹ pairs each | Decremental counter, fixed prefix — static test-vector key |
| `dataset5`–`dataset7` | ~2¹⁶·³ / 2¹⁴·⁶ / 2¹⁵·³ pairs | 8-byte blocks read from `pdfs/1.pdf`–`3.pdf`, deduped, **random 80-bit key per dataset** |

## Model Architecture

The baseline model ([`model.py`](model.py)) is a stacked **LSTM**
regressor:

```
Input (8, 1)
 → LSTM(64, tanh, return_sequences)
 → Dropout(0.5)
 → LSTM(64, tanh)
 → Dropout(0.5)
 → Dense(8, linear)          # predicts the 8 plaintext bytes
```

The training scripts use a related stacked-LSTM variant (`tanh` → `sigmoid` → `relu`
with `SpatialDropout1D`), trained with MAE loss and Adam/RMSprop optimizers. Per-round
hyperparameters (units, learning rate, epochs, optimizer) are taken from the reference
paper where available.

## Results

Round-by-round metrics are stored in [`results/`](results/). As expected, **restoration
accuracy decays as the number of rounds increases** — the cipher becomes progressively
harder to invert.

### CTT (dataset1) — selected rounds

| Round | Optimizer | Test Accuracy | Mean Byte Error |
|-------|-----------|---------------|-----------------|
| 1     | Adam      | 75.58%        | 0.2125          |
| 5     | RMSprop   | 74.43%        | 0.2290          |
| 15    | RMSprop   | 71.91%        | 0.2893          |
| 20    | Adam      | 70.96%        | 0.3299          |
| 31    | RMSprop   | **68.04%**    | 0.3905          |

Full tables for all 31 rounds and all three regimes are in
`results/ctt_round_results.csv`, `results/nctt_round_results.csv`, and
`results/rwtt_round_results.csv`.

## Notes & Caveats

- The training scripts have a **primary path** (which expects helper modules such as
  `model.train_for_all_round` / `utils` that are not committed) and a **fallback path**
  that runs the round-by-round loop directly. In practice the fallback loop is what
  produces the committed results and `.h5` models.
- The loader (`preprocessing.load_all_data`) reads from `src/data/`; keep that folder in
  sync with `datasets/`.
- A few scripts contain minor issues (e.g., a typo in one metric helper, and a printed
  `Hamming_Accuracy` key that isn't populated in the result dict). Review the round loop
  in each `train_*.py` before relying on exact reproductions.
- The `pdfs/` are the raw material for the real-world datasets and, in this repo, also
  serve as the reference material for the methodology.

## References

- A. Bogdanov et al., *"PRESENT: An Ultra-Lightweight Block Cipher"*, CHES 2007.
- The reference paper(s) and background material for this experiment are included under
  [`pdfs/`](pdfs/).

---

*This README documents the repository as studied from its source. Acronym expansions
(CTT/NCTT/RWTT) follow the repo's own "correlated / non-correlated / real-world"
terminology; consult the papers in `pdfs/` for the authors' exact definitions.*

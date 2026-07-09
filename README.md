# Low-Dose CT Denoising with RED-CNN

Denoises low-dose ("quarter-dose") CT scans toward normal-dose quality, using RED-CNN (Chen et al.,
2017) trained on the 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge dataset.

This is an image-domain denoising project, not sinogram/projection-domain reconstruction — see
"Key design decisions" below for why, along with a summary of every other choice made and rejected.
Full reasoning for each lives in the notebook itself, next to the code it explains.

## Results

| | PSNR | SSIM |
|---|---|---|
| Low-dose input (no model) | 17.02 dB | 0.673 |
| RED-CNN output | 22.27 dB | 0.774 |

Measured on a held-out test set (2 patients, 1,136 slices) not seen during training.

**Before reading those numbers as good or bad:** PSNR here is computed inside a 400 HU display window
appropriate for abdominal CT. That choice alone can shift PSNR by around 17 dB relative to a paper
using a wider window on the same underlying image quality — confirmed directly during this project,
not assumed. A published RED-CNN PSNR of ~32 dB on this same dataset is not a fair comparison unless
its exact normalization is known. The relative gain over this project's own baseline is the meaningful
number, not the absolute one — full reasoning is in the notebook's critical discussion.

## Key design decisions

Full reasoning for these, and every other decision made on this project, is in the notebook's design
decisions and lessons-learned sections — this is a summary for anyone not planning to open it.

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Data source | Raw DICOM | Pre-processed PNGs also included in the dataset | Traced several PNGs back to their source DICOM and found the pixel scaling is per-scan adaptive, not a fixed window — the same tissue value maps to a different pixel intensity depending on which patient it came from |
| Model | RED-CNN (plain CNN) | GAN (WGAN-VGG style) | A fully-understood CNN, including its known over-smoothing limitation, beats a GAN result that isn't fully explainable on a first attempt |
| Learning rate | 1e-4 (the paper's original value) | 5e-4, briefly, based on a reduced-scale test | A quick sweep favored 5e-4; a full-scale run showed that was a scale artifact and 1e-4 was fine all along — kept as a documented reversal, not quietly corrected |
| `torch.compile` / `channels_last` | Neither | Both were suggested more than once | Confirmed compatibility risk on Pascal GPUs for the former, a confirmed silent-correctness bug on `ConvTranspose2d` for the latter — not left out by oversight |
| Train/val/test split | One fixed 7/1/2 patient split | Leave-one-patient-out cross-validation (the paper's method) | ~10x cheaper to run; a stated rigor tradeoff, not an unstated one |

Also worth knowing going in:
- Trained with plain MSE loss, which is known to favor smoothness over fine detail — no perceptual or
  adversarial loss term here, a deliberate scope decision, not an oversight.
- The "low dose" scans are simulated (Poisson noise added to full-dose projection data), not
  independent low-dose acquisitions of the same patient.

## Repository structure

```
.
├── notebooks/
│   └── red_cnn_training.ipynb   data loading, model, training, evaluation -- and the full
│                                 reasoning behind every decision, in markdown cells next to
│                                 the code each one explains
├── requirements.txt
└── README.md
```

## Running it

Built for and tested on [Kaggle Notebooks](https://www.kaggle.com/code) with a GPU accelerator
attached, using the [`ct-low-dose-reconstruction`](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction)
dataset attached to the session. The dataset itself isn't bundled in this repo — check its license on
the Kaggle page before using it beyond personal experimentation.

1. Open `notebooks/red_cnn_training.ipynb` in a Kaggle Notebook (or locally, with the dataset
   downloaded and `ORIG_ROOT` in the notebook pointed at it).
2. Attach the dataset above, with a GPU accelerator enabled.
3. Run all cells. `SMOKE_TEST = True` restricts to a tiny subset for a ~1-2 minute sanity check before
   committing to a full run; the notebook ships with `SMOKE_TEST = False`.
4. A full run (7 training patients, 30 epochs) takes roughly 50-60 minutes on a T4/P100.

To run outside Kaggle, install the dependencies below and point `ORIG_ROOT` at wherever the dataset's
`Original Data` folder lives locally.

```
pip install -r requirements.txt
```

## Architecture

RED-CNN: 5 convolutional + 5 deconvolutional layers, symmetric, no pooling, three residual shortcut
connections. ~1.85M parameters. Full description, including the shortcut wiring (verified against a
public reference implementation, not just the paper's prose) and a known limitation of this
architecture's small receptive field, is in the notebook.

## Engineering notes

A meaningful part of this project was debugging real problems rather than a clean first run —
Kaggle's dataset mount path shifting shape mid-project, a learning-rate scheduler bug that looked like
a training plateau, a caching bug that made a 3-epoch test take 30 minutes, and a hyperparameter sweep
that gave a different answer at full scale than it did at reduced scale. All of it is documented in the
notebook's "Lessons learned" section rather than smoothed over, since that's a more honest — and more
useful — record than a notebook that only shows the parts that worked on the first try.

## License

Code in this repository is MIT licensed (see [LICENSE](LICENSE)). The CT imaging dataset used here is
not included and is governed by its own license on Kaggle — check that page before redistributing or
using it beyond personal experimentation.

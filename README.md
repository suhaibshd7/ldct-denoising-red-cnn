# Low-Dose CT Denoising with RED-CNN

Denoises simulated quarter-dose CT slices toward full-dose quality using RED-CNN (Chen et al., 2017, [arXiv:1702.00288](https://arxiv.org/abs/1702.00288)), trained on the [2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge dataset](https://www.aapm.org/GrandChallenge/LowDoseCT/) (10 patients, 1mm slice thickness, sharp/D45 reconstruction kernel).

CT relies on ionizing radiation, and lower dose means more quantum noise in the reconstructed image. This project trains an image-domain denoiser to recover full-dose-like quality from quarter-dose input, and evaluates it against both a do-nothing baseline and two classical (non-learned) denoisers, rather than only against itself.

**Scope note:** this works in the image domain (denoising already-reconstructed DICOM slices), not the sinogram/projection domain , raw projection data isn't available here, and image-domain denoising is the problem a downstream developer without scanner access can actually work on.

## Results

Held-out test set: 2 patients, 1136 slices, never seen during training or hyperparameter selection.

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Low-dose input (no processing) | 16.61 ± 1.58 | 0.747 ± 0.049 |
| Gaussian filter (σ=1) | 20.82 ± 1.39 | 0.726 ± 0.052 |
| Non-local means (skimage, default params) | 20.17 ± 1.40 | 0.751 ± 0.056 |
| **RED-CNN (this model)** | **22.27 ± 1.58** | **0.774 ± 0.049** |

*(Classical-baseline row and RED-CNN row above are computed on the same 200-slice matched subsample, for a fair comparison , non-local means is slow enough that running it on all 1136 test slices wasn't practical. RED-CNN's own full-test-set number, 1136 slices, is 22.27 ± 1.57 dB / 0.774 ± 0.049 , consistent with the subsample.)*

**RED-CNN clearly beats both classical baselines, but not by the same margin on both metrics.** It beats the better classical method on PSNR (Gaussian, 20.82 dB) by 1.45 dB, and beats the better classical method on SSIM (NLM, 0.751) by 0.023 , NLM in particular gets respectably close on structural similarity even though it's well behind on PSNR.

**The Gaussian filter result is worth sitting with:** it raises PSNR substantially (16.6 → 20.8 dB) while making SSIM *slightly worse than doing nothing at all* (0.726 vs 0.747 raw). That's a real, measured example of PSNR and SSIM disagreeing, and of naive smoothing improving one metric while degrading structural fidelity , not a hypothetical caveat, an actual number from this run.

### Per-patient breakdown

Pooling both test patients into one mean hides real between-patient variation, so results are reported separately:

| Patient | Slices | PSNR: input → RED-CNN | SSIM: input → RED-CNN | PSNR gain |
|---|---|---|---|---|
| L333 | 610 | 15.89 → 21.78 dB | 0.719 → 0.749 | +5.89 dB |
| L506 | 526 | 17.42 → 22.83 dB | 0.777 → 0.802 | +5.42 dB |

The two test patients differ by 1.53 dB in *baseline* noise level before any processing , a reminder that "the test set" here means two specific scans, not a population sample.

### What can and can't be claimed statistically

With only 2 independent test patients, there isn't a statistically valid way to put a confidence interval or p-value on "this generalizes to a new patient" , that would need more held-out patients than a 10-patient dataset can spare without shrinking the training set further. The per-slice consistency *within* each patient's scan is high (slice-level PSNR gain: 5.67 ± 0.39 dB, pooled), which shows the improvement isn't driven by a handful of easy slices , but that's a statement about within-scan consistency, not about between-patient generalization. Treating all 1136 slices as independent samples for a population-level test would be a real statistical error (pseudo-replication): slices from the same scan share anatomy and noise characteristics, so the true number of independent units for a generalization claim is 2, not 1136.

### Qualitative check: where does it do worst?

The three lowest-PSNR test slices after denoising (18.09–18.22 dB, vs. a 22.27 dB mean) are all from the same patient (L506) and are nearly adjacent slice indices , suggesting a localized anatomical region rather than scattered random failures. See `comparison.png` for the actual images (typical slice vs. this worst region, at each of low-dose / RED-CNN / full-dose).

Visual inspection of the worst slice (index 691, an abdominal cross-section showing liver, spleen, and kidneys) shows a specific, recognizable failure mode rather than generic degradation: the raw low-dose input here is unusually noisy even by this dataset's standards, and while RED-CNN still reduces the graininess substantially, it does so by over-smoothing , internal liver/spleen texture and vessel boundaries that are sharp in the full-dose target come out blurred in the RED-CNN output. By contrast, the typical-case slice (pelvis/femoral heads) shows the model recovering anatomy cleanly with only mildly softer texture than the native full-dose image. This is a direct illustration of the MSE-over-smoothing limitation noted below: the noisier the input, the more the model leans on aggressive smoothing to minimize MSE loss, at the cost of fine structural detail , precisely the failure mode a perceptual or adversarial loss term would be expected to help with.

## Design decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Scope | Image-domain denoising | Sinogram-domain reconstruction | Raw projection data isn't available; denoising is the tractable, actually-available problem |
| Model | RED-CNN (plain CNN) | GAN (e.g. WGAN-VGG) | A fully-understood CNN, including its known over-smoothing limitation, is preferable to a GAN result that isn't fully explainable on a first attempt |
| Data source | Raw DICOM | Preprocessed PNGs in the Kaggle mirror | The preprocessed PNGs use per-scan adaptive scaling, not a fixed window , confirmed empirically, not assumed |
| HU window | Clip [-160, 240], then min-max to [0,1] | Wide/unclipped range, z-score standardization | The standard abdominal soft-tissue window (WL 40 / WW 400); also why absolute PSNR here reads lower than papers using a wider window (see below) |
| Slice thickness / kernel | 1mm, Sharp (D45) | 3mm, Soft (B30) | More slices per patient; D45's noise is closer to independent per-pixel, matching an MSE loss's assumptions; 3mm Full Dose was also missing for all 10 patients in this dataset mirror |
| Patch strategy | Random 64×64 crops, on the fly | Original paper's 55×55 fixed sliding window | Simpler to implement; a documented simplification also used in public reimplementations |
| Patch augmentation | None beyond random cropping | Paper's rotation (45°) + flip + scaling (0.5–2×) | Axial CT slices are always acquired in a fixed, canonical orientation , a 45°-rotated patch is geometry the model never sees at inference, and scaling doesn't fit a fixed-resolution denoising task. Flip is more defensible (mirrored anatomy is still broadly plausible) but was left out too, for simplicity; cost of that tradeoff is in Known limitations |
| Train/val/test split | Fixed patient split (7/1/2) | Full leave-one-patient-out cross-validation (as in the original paper) | ~10x cheaper to run; appropriate for a portfolio-scale project, explicitly logged as a rigor tradeoff, not an oversight |
| Learning rate | 1e-4, tested rather than assumed | 5e-4, 1e-3 | A reduced-scale sweep initially favored higher rates, but a full 30-epoch run showed that was a scale artifact , higher rates were unstable, and converged within 0.02 dB of 1e-4 anyway |
| Classical baselines | Gaussian (σ=1), NLM (skimage defaults) | Hand-tuned NLM / BM3D | Sanity floors ("does the model clearly beat a standard method"), not a search for the best possible classical result |
| Performance code | cuDNN autotune, fused Adam, GPU-side loss accumulation | pinned memory, persistent workers, prefetch, batched evaluation | These three measurably cut training time on ~16,000 training steps; the rest don't , the dataset is already preloaded into RAM as small (~1MB/batch) tensors, so there's no meaningful transfer cost to overlap, and evaluation's real cost is skimage's CPU-side SSIM computation, not the GPU forward pass |

## Model

RED-CNN: 10 layers (5 conv + 5 deconv), 5×5 kernels, stride 1, no padding, no pooling, ~1.85M parameters. No padding + matching kernel sizes means output size always equals input size (traced by hand: 64→60→56→52→48→44 through the conv layers, then back up 44→48→52→56→60→64 through the deconv layers , which is also why the shortcut connections land on matching tensor shapes). Three additive shortcuts (conv2→before deconv4, conv4→before deconv2) plus a global residual (input added to final output), following a verified public reference implementation since the original paper's text doesn't fully specify the wiring.

**Known architectural limitation:** 5 layers of 5×5 kernels with no pooling gives a receptive field of ~21×21 pixels , a fairly small spatial context, and a plausible contributor to the early training plateau (validation PSNR is nearly flat from epoch ~15 onward), independent of learning rate.

## Known limitations

- **No patch augmentation is used**, unlike the original paper's 45° rotation, flip, and scaling on training patches (see Design decisions above for why rotation/scaling specifically don't fit this data). With only 7 training patients, skipping augmentation altogether plausibly costs real effective training diversity , that cost hasn't been measured, so its size is unknown rather than ruled out.
- **Validation is 1 patient** (L310, 533 slices). Checkpoint selection and the learning-rate comparison during development were both effectively tuned against that one patient's anatomy. Worth noting concretely: an earlier version of this evaluation checked validation on only the first 30 slices of that patient as a shortcut, which gave a baseline PSNR of 17.02 dB , evaluating the *full* 533-slice validation set instead gives 14.75 dB, a 2.3 dB difference. That first-30-slices subset was measurably easier than the full scan. The numbers in this README use the full validation set.
- **Train/val/test is a single fixed split**, not the leave-one-patient-out cross-validation used in the original paper , a deliberate cost/rigor tradeoff for a 10-patient dataset, not an oversight.
- **Classical baselines are sanity floors**, not exhaustively tuned best cases , Gaussian σ=1 and NLM with skimage's documented rule-of-thumb starting parameters, not a grid search.
- **"Quarter dose" is simulated** (Poisson noise added to full-dose projection data), not an independent lower-dose acquisition of the same patient , results are somewhat idealized relative to real deployment.
- **MSE loss is known to over-smooth fine structure**, and PSNR/SSIM don't fully capture diagnostic usefulness. This run has a direct example of the two metrics disagreeing: the Gaussian-filter result above improves PSNR while slightly *hurting* SSIM relative to doing nothing.
- **Small receptive field** (~21×21 pixels) is a plausible structural limit on how much anatomical context the model can use, independent of any training choice.
- **2 independent test patients** means no statistically valid claim about between-patient generalization can be made from this evaluation alone (see "What can and can't be claimed statistically" above).

What a more rigorous version would add: a perceptual or adversarial loss term to address over-smoothing; full leave-one-patient-out cross-validation; a downstream task check (e.g., does denoising change measured lesion size/density) rather than only pixel-level metrics; independent low-dose acquisitions rather than simulated ones; a wider architecture search; and testing whether flip augmentation alone recovers some of the lost training diversity, given rotation and scaling are excluded on principle.


## Requirements

`torch`, `pydicom`, `numpy`, `pandas`, `scikit-image`, `matplotlib`. Run with the [`andrewmvd/ct-low-dose-reconstruction`](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction) dataset attached (Kaggle). `pydicom` is not always preinstalled on Kaggle's GPU image.

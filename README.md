# Low-Dose CT Denoising with RED-CNN

Denoises low-dose ("quarter-dose") CT scans toward normal-dose quality using a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN; Chen et al., 2017) trained on the 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge dataset.

Computed tomography relies on ionizing radiation, and cumulative exposure carries long-term risks. Accordingly, clinical practice is strictly guided by the ALARA principle: **"as low as reasonably achievable."** While lowering the radiation dose directly reduces patient risk, fewer photons reaching the detector create severe quantum noise in the reconstructed image, potentially masking subtle diagnostic features.

AI-based image-domain denoising breaks this traditional tradeoff. By computationally restoring low-dose scans close to full-dose quality, it offers a scalable software pathway toward safe imaging. This repository implements that end-to-end processing pipeline on real clinical patient data—not as a simplified toy exercise, but facing the full engineering complexity of raw DICOM structures.

> **Scope Note:** This project operates entirely within the **image-domain** (denoising already-reconstructed slices), rather than the sinogram/projection-domain. Projection data is rarely available to downstream developers in clinical environments, making image-domain denoising the most practical and impact-ready problem to solve.

---

## Results

Measured on a completely held-out test set consisting of **2 patients (1,136 distinct clinical slices)** unseen during training or hyperparameter tuning.

| Configuration | PSNR ↑ | SSIM ↑ |
| :--- | :---: | :---: |
| **Low-Dose Input** (No Model Baseline) | 17.02 dB | 0.673 |
| **RED-CNN Output** | **22.27 dB** | **0.774** |

* **PSNR (Peak Signal-to-Noise Ratio):** Measures noise level relative to the reference image (higher is cleaner).
* **SSIM (Structural Similarity Index Measure):** Ranges from 0 to 1; closer to 1 indicates better structural and feature retention.

### A Critical Note on Metric Interpretation
When comparing these numbers to external literature, **context is everything.** PSNR in this project is calculated strictly inside a **400 HU display window** optimized for abdominal CT evaluation. This localized constraint can drop absolute PSNR values by approximately 17 dB compared to papers calculating metrics across a wider, unclipped HU spectrum on the exact same images. 

A published RED-CNN paper reporting ~32 dB on this dataset cannot be compared with these figures unless their exact normalization rules are mirrored. The relative improvement over the unmitigated baseline ($\Delta\text{PSNR} = +5.25\text{ dB}$) is the only scientifically rigorous baseline of success.

---

## Comprehensive Design Decisions & Defenses

Every architectural, data, or pipeline optimization choice carried distinct tradeoffs. Below is the complete design log summarizing what was chosen, rejected, and why.

| Decision | Chosen | Rejected | Why |
| :--- | :--- | :--- | :--- |
| **Data Source** | **Raw DICOM (`Original Data`)** | Pre-processed PNGs bundled in the Kaggle mirror | Empirically verified that the preprocessed PNG paths utilize *per-scan adaptive scaling* rather than a fixed window. This means identical raw tissue values map to wildly different pixel intensities across patients. Raw DICOMs protect true Hounsfield Units (HU). |
| **Target Architecture** | **RED-CNN** | Generative Adversarial Networks (WGAN-VGG) | For initial deployment, a fully-understood CNN with mathematically determinable bounds—even with its known smoothing artifacts—is preferred over non-deterministic GAN hallucinations that cannot be clinically explained. |
| **CT Slice Thickness** | **1mm Slices** | 3mm Slices | Maximizes the slice sample pool per patient. Furthermore, 3mm full-dose reference slices were corrupted or completely missing for all 10 patient directories in this dataset mirror. |
| **Reconstruction Kernel** | **Sharp Kernel (D45)** | Soft Kernel (B30) | Noise artifacts under the D45 sharp kernel exhibit near-random, pixel-independent distributions, aligning properly with the assumptions of an element-wise MSE loss. |
| **HU Window Optimization** | **Clip `[-160, 240]` $\to$ Min-Max to `[0,1]`** | Z-score standardization | Isolates the clinically relevant window for abdominal imaging while cleanly dropping extreme HU outliers driven by contrast agents or dense metallic implants. |
| **Patch Strategy** | **On-the-fly random $64\times64$ crops** | Original $55\times55$ regular sliding window | Drastically simplifies the dataset implementation footprint. On-the-fly randomization acts as a robust regularizer, preventing location-bound overfitting. |
| **Data Loading Loop** | **Preload all slices directly to RAM** | Lazy-load from disk + LRU cache | The full training set (~4,300 paired arrays) fits cleanly within standard 16GB system RAM (~8-9GB footprint). Caching arrays in system RAM eliminates disk I/O bottlenecks and keeps multi-worker threading safe. |
| **Learning Rate Selection** | **$1\times 10^{-4}$ (Tested)** | $5\times 10^{-4}$ or $1\times 10^{-3}$ | Literature varies wildly on RED-CNN hyperparameters ($1\times10^{-5}$ to $5\times10^{-4}$). Localized sweeps initially favored higher rates, but full-scale training proved those gains were scale artifacts. $1\times10^{-4}$ proved stable over 30 epochs. |
| **Validation Flow** | **Store metric scalars only** | Keeping complete test/validation arrays in memory | Storing multi-gigabyte arrays merely for occasional diagnostic plots causes massive memory pressure. Slicing predictions into individual calls for plotting (`predict_one`) keeps memory flat. |
| **Validation Splitting** | **Fixed Patient Partition (7/1/2)** | Full Leave-One-Patient-Out Cross-Validation | Decreases overall training execution costs by an order of magnitude ($\sim10\times$). Explicitly logged as an engineering runtime tradeoff rather than a hidden shortcut. |

---

## Technical Architecture

The implemented RED-CNN network contains **10 layers symmetrically paired** (5 Convolutional layers, 5 Convolution-Transpose/Deconvolutional layers) with approximately **1.85 Million trainable parameters**:

* **Properties:** All hidden layers leverage $5\times5$ kernels, a stride of 1, and no interior padding or pooling operations. Omitting pooling avoids discarding fine structural variations crucial for radiology diagnoses.
* **Spatial Independence:** Because layers lack spatial pooling, the output dimension matches the input dimension exactly. The exact same model weights train on $64\times64$ patches but seamlessly accept full $512\times512$ patient slices during validation and inference.
* **Shortcut Wiring:** Contains three symmetric additive shortcut connections alongside a global identity shortcut matching the paper's functional requirements:
  $$\text{Conv2 Output} \longrightarrow \text{Deconv4 Input}$$
  $$\text{Conv4 Output} \longrightarrow \text{Deconv2 Input}$$
  $$\text{Global Input Image} \longrightarrow \text{Final Output Layer}$$
* **Known Limitations:** Because it lacks pooling, 5 layers of $5\times5$ kernels restrict the effective receptive field to a narrow window ($\sim21\times21$ pixels). This limited spatial context causes the network to struggle with macro-structural reconstruction, contributing directly to the performance ceiling of pure MSE-driven networks.

---

## Performance & Engineering Notes

Several common deep learning optimization techniques were evaluated and **deliberately excluded** from production. In medical image reconstruction, preserving correctness and avoiding code overhead outweighs marginal speedups:

* **`torch.compile`:** Disabled. An outstanding PyTorch bug affecting older Pascal-architecture GPUs (such as the NVIDIA P100 often assigned by cloud notebooks) breaks backend Triton/TorchInductor generation. Disabling compilation guarantees cross-hardware reproducibility.
* **`channels_last` Memory Layout:** Disabled. PyTorch possesses confirmed bugs where switching to `channels_last` produces **silently incorrect outputs** inside `ConvTranspose2d` layers. Because RED-CNN uses 5 of these layers, a speedup that sacrifices tensor integrity cannot be tolerated.
* **Automatic Mixed Precision (AMP):** Omitted. AMP speeds up runtime on Tensor-Core cards but adds substantial code overhead via gradient scaling (`GradScaler`) and autocasting, clouding the architectural readability of the training loop for educational use.
* **VRAM Preloading:** Disabled. Moving the entire training array into GPU VRAM pushes standard 16GB allocation limits once gradients and optimizer states inflate, posing an extreme Out-of-Memory (OOM) crash risk.

### Retained Optimizations
* `torch.backends.cudnn.benchmark = True`: Safely active since all extracted image patches match a fixed $64\times64$ geometry.
* `fused=True` on the Adam optimizer: Minimizes GPU kernel launch steps on CUDA platforms.
* Validation passes are fully batched rather than looped slice-by-slice, providing high throughput while preserving bitwise numerical equivalence.

---

## Repository Structure

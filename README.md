# Deep Variance Stabilization + Diffusion Denoising for Non-Stationary Rician MRI Noise

<p align="center">
  <img src="vst_denoiser_framework.png" width="900"/>
</p>

---

## Overview

Magnetic Resonance Imaging (MRI) magnitude images are corrupted by **Rician-distributed noise**. Classical MRI denoisers typically assume **stationary** noise, which holds approximately for ideal single-coil acquisitions.

However, in real clinical scanners with:

- multi-coil receivers
- coil sensitivity maps
- GRAPPA/SENSE reconstructions
- partial k-space acquisitions

the noise becomes **spatially varying**, leading to **non-stationary Rician noise**. This breaks assumptions made by denoisers and diffusion models trained on IID Gaussian noise.

To resolve this, we propose a **two-stage denoising pipeline**:

> **Dataset → SigmaNet → VSTNet → Diffusion → Final MRI**

---

## Pipeline Summary

### Stage I — Variance Stabilization (Physics-Informed)

1. **SigmaNet** predicts spatial noise variance
2. **VSTNet** transforms non-stationary Rician → stationary Gaussian

\[
I(x) \sim \text{Rician}(\mu(x), \sigma^2(x)) \Rightarrow \tilde{I}(x) \sim \mathcal{N}(\mu, 1)
\]

### Stage II — Diffusion Denoising

Diffusion denoisers operate on IID Gaussian corrupted images:

\[
\tilde{I} = x_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
\]

This **decouples physical noise modeling** from learned denoising and enables diffusion models to generalize.

---

## Key Contributions

✔ Handles **non-stationary Rician** MRI noise  
✔ Converts MRI noise to **IID Gaussian** for diffusion  
✔ Improves SNR and anatomical detail preservation  
✔ Modular: DDPM, DDIM, Score-SDE are interchangeable  
✔ Validated on phantom + simulated MRI datasets  

---

# 🔧 Pipeline Steps + Commands

---

## 1. Synthetic Non-Stationary Rician Dataset

```bash
python syn_non_stat_rician_add.py \
  --in_dir <clean_mri_folder> \
  --out_dir <output_dataset_folder> \
  --n_aug 1 \
  --sigma_map_mode radial \
  --sigma_blur 25.0 \
  --percentNoise 11.0 \
  --seed 0 \
  --save_png \
  --save_clean_npy
Produces:
clean_npy/
noisy_npy/
sigma_npy/
noisy_png/
sigma_png/
manifest.csv
# 2. Train SigmaNet (Noise Variance Estimator)
python train_homomorphic_sigmanet.py \
  --manifest <manifest.csv> \
  --base_dir <root_dir> \
  --out_dir <sigmanet_ckpts> \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --wd 1e-4 \
  --val_split 0.1 \
  --blur_ksize 31 \
  --blur_sigma 7.0 \
  --base 32 \
  --input_mode h_only \
  --tv_weight 0.05 \
  --max_viz 3 \
  --amp
# 3. SigmaNet Inference
python infer_homomorphic_sigmanet.py \
  --ckpt <sigmanet_best_or_last.pth> \
  --manifest <manifest.csv> \
  --base_dir <root_dir> \
  --out_dir <sigma_predictions> \
  --input_mode h_only \
  --blur_ksize 31 \
  --blur_sigma 7.0 \
  --eval_gt

Produces: 
sigma_pred_npy/
sigma_pred_png/
metrics.csv
# 4. Train VSTNet (Variance Stabilizer)
python train_vstnet_fixed.py \
  --out_dir <vstnet_runs> \
  --synthetic 0 \
  --noisy_dir <noisy_png> \
  --sigma_dir <sigma_npy> \
  --sigma_kind npy \
  --sigma0_is_variance 0 \
  --pad_multiple 8 \
  --device cuda \
  --epochs 20 \
  --batch_size 8

# 5. VSTNet Inference → Generate Stabilized Images
python infer_vstnet_fixed.py \
  --ckpt <vstnet_ckpt.pt> \
  --noisy_dir <noisy_png> \
  --sigma_dir <sigma_npy> \
  --sigma_kind npy \
  --sigma0_is_variance 0 \
  --pad_multiple 8 \
  --out_dir <vst_output> \
  --device cuda \
  --batch_size 8 \
  --num_workers 2 \
  --log_every 10 \
  --use_snr_proxy 0 \
  --blur_ks 21 \
  --blur_sigma 3.0 \
  --save_extra 0
Outputs:
I_tilde_npy/   <-- used for diffusion training
I_tilde_png/
u1_u2.csv
# 6. Diffusion Training (Stage II)
python mri_denoiser_controlled_noise.py \
  --mode train \
  --image_size 0 \
  --train_clean <clean_folder> \
  --train_batch_size 16 \
  --train_steps 100000 \
  --timesteps 1000 \
  --save_dir <output_dir>
# 7. Diffusion Inference
python mri_denoiser_controlled_noise.py \
  --mode denoise_few \
  --image_size 0 \
  --weights <checkpoint>.pt \
  --noisy_folder <input_folder> \
  --out_folder <output_folder> \
  --few_steps 20 \
  --eta 0.0
Why Non-Stationary MRI Noise is Difficult

Traditional single-coil MRI is often modeled with stationary noise, meaning noise variance does not change spatially:

σ(x) = σ0

However, modern MRI uses multiple receiver coils, coil sensitivity profiles, and parallel imaging, which produce non-stationary Rician noise:

I(x) ~ Rician( μ(x), σ²(x) )


# Repository Structure
├── syn_non_stat_rician_add.py
├── train_homomorphic_sigmanet.py
├── infer_homomorphic_sigmanet.py
├── train_vstnet_fixed.py
├── infer_vstnet_fixed.py
├── mri_denoiser_controlled_noise.py
├── vst_denoiser_framework.png
└── README.md

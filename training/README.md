# Instruction for Training

Once all the datasets and pretrained models are prepared, we can lauch the training jobs on three benchmarks 
using the scripts under the `scripts` folder. Note that ImageNet is always required for each benchmark.
Please change the paths accordingly in the scripts.

#### Training on PF-PASCAL

```bash
bash scripts/train_moco_icycle_pfpascal_aug_attention_relu-gpu_8-lr_0.003-bs_256_layer_13-downscale_16-drop_0-icycle_lw_0.0005-temp_0.0007-lc_0.sh
```

#### Training on PF-WILLOW

```bash
bash scripts/train_moco_icycle_pfwillow_aug_attention_relu-gpu_8-lr_0.003-bs_256_layer_13-downscale_16-drop_0-icycle_lw_0.00005-temp_0.0007-lc_0.sh
```

#### Training on SPair-71k

```bash
bash scripts/train_moco_icycle_spair_aug_attention_relu-gpu_8-lr_0.003-bs_256_layer_13-downscale_16-drop_0-icycle_lw_0.0005-temp_0.0007-lc_0.sh
```

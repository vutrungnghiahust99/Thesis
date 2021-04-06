Experiments using eriklindernoren architecture
---
commit: "to HDD"

plot
```bash
./plots/eriklindernoren_architecture/plot.sh
```
---
Experiment using SAGAN architecture with conv_dim=64
---
```bash
./runs/sagan_architecture/run_spectral_norm_lsgan.sh
./runs/sagan_architecture/run_spectral_norm_gan1.sh
```
Experiment using SAGAN architecture with conv_dim=16
---
```bash
./runs/sagan_architecture/run_spectral_norm_lsgan_d_conv_dim=16.sh
./runs/sagan_architecture/run_spectral_norm_gan1_d_conv_dim=16.sh
```
Experiment using SAGAN architecture with conv_dim=16 and using attention
---
```bash
./runs/sagan_architecture/run_spectral_norm_lsgan_d_conv_dim=16_attention.sh
./runs/sagan_architecture/run_spectral_norm_gan1_d_conv_dim=16_attention.sh
```
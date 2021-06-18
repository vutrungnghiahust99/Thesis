Enviroments
---
- pytorch
- pandas

Reproduce experimental results
---
Firstly, navigate to the project
```bash
cd rfgan
```
All commands to reproduce experimental results are saved in `./run/` folder.

Below is an example to run the experiment:
 - RFGan + No augmentation + No bagging + 10 heads + eriklindernoren architecture
```bash
python -m src.train.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation0_10heads_usemaskD \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '453' \
                          --interval '4' \
                          --use_mask_d '1'
```

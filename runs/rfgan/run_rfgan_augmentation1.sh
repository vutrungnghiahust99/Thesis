# Do not use bagging
python -m src.train.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation1_10heads_usemaskD \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '453' \
                          --interval '4' \
                          --use_mask_d '1' \
                          --augmentation '1' \
                          --aug_times '2'

# use bagging
python -m src.train.rfgan --exp_name rfgan_lossgan1_bagging1_augmentation1_10heads_usemaskD \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '453' \
                          --interval '4' \
                          --bagging '1' \
                          --use_mask_d '1' \
                          --augmentation '1' \
                          --aug_times '2'

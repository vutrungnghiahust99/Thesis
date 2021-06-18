# Do not use bagging
python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation0_3heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '3' \
                                --n_epochs '121' \
                                --interval '4' \
                                --use_mask_d '1'

python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation0_5heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '5' \
                                --n_epochs '121' \
                                --interval '4' \
                                --use_mask_d '1'

python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation0_7heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '7' \
                                --n_epochs '121' \
                                --interval '4' \
                                --use_mask_d '1'

# use bagging
python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging1_augmentation0_3heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '3' \
                                --n_epochs '121' \
                                --interval '4' \
                                --bagging '1' \
                                --use_mask_d '1'

python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging1_augmentation0_5heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '5' \
                                --n_epochs '121' \
                                --interval '4' \
                                --bagging '1' \
                                --use_mask_d '1'

python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging1_augmentation0_10heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '10' \
                                --n_epochs '121' \
                                --interval '4' \
                                --bagging '1' \
                                --use_mask_d '1'

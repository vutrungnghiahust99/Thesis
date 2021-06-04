# Do not use bagging
python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation0_5heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '5' \
                                --n_epochs '453' \
                                --interval '4' \
                                --use_mask_d '1'

# use bagging
python -m src.train_sagan.rfgan --exp_name rfgan_lossgan1_bagging1_augmentation0_5heads_usemaskD_sagan \
                                --loss_name 'gan1' \
                                --n_heads '5' \
                                --n_epochs '453' \
                                --interval '4' \
                                --bagging '1' \
                                --use_mask_d '1'

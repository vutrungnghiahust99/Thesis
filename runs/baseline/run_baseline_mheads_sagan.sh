# Do not use augmentation
python -m src.train_sagan.mdgan --exp_name 'baselinemheads_lossgan1_5heads_augmentation0_sagan' \
                                --loss_name 'gan1' \
                                --n_heads '5' \
                                --n_epochs '453' \
                                --interval '4'

# Use augmentation
python -m src.train_sagan.mdgan --exp_name 'baselinemheads_lossgan1_5heads_augmentation1_sagan' \
                                --loss_name 'gan1' \
                                --n_heads '5' \
                                --n_epochs '453' \
                                --interval '4' \
                                --augmentation '1' \
                                --aug_times '2'

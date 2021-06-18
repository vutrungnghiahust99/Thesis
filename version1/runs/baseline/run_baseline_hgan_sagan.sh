# Do not use augmentation
python -m src.train_sagan.hgan --exp_name 'baselinehgan_losshgan_5heads_augmentation0_sagan' \
                               --n_heads '5' \
                               --n_epochs '121' \
                               --interval '4'

# Use augmentation
python -m src.train_sagan.hgan --exp_name 'baselinehgan_losshgan_5heads_augmentation1_sagan' \
                               --n_heads '5' \
                               --n_epochs '121' \
                               --interval '4' \
                               --augmentation '1' \
                               --aug_times '2'

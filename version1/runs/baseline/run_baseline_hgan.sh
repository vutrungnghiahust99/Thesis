# Do not use augmentation
python -m src.train.hgan --exp_name 'baselinehgan_losshgan_augmentation0' \
                         --n_heads '10' \
                         --n_epochs '453' \
                         --interval '4'

python -m src.train.hgan --exp_name 'baselinehgan_losshgan_5heads_augmentation0' \
                         --n_heads '5' \
                         --n_epochs '453' \
                         --interval '4'

python -m src.train.hgan --exp_name 'baselinehgan_losshgan_3heads_augmentation0' \
                         --n_heads '3' \
                         --n_epochs '453' \
                         --interval '4'

# Use augmentation
python -m src.train.hgan --exp_name 'baselinehgan_losshgan_augmentation1' \
                         --n_heads '10' \
                         --n_epochs '453' \
                         --interval '4' \
                         --augmentation '1' \
                         --aug_times '2'

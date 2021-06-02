# Do not use augmentation
python -m src.train.mdgan --exp_name 'baselinemheads_lossgan1_augmentation0' \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '453' \
                          --interval '4'

python -m src.train.mdgan --exp_name 'baselinemheads_lossgan1_5heads_augmentation0' \
                          --loss_name 'gan1' \
                          --n_heads '5' \
                          --n_epochs '453' \
                          --interval '4'

python -m src.train.mdgan --exp_name 'baselinemheads_lossgan1_3heads_augmentation0' \
                          --loss_name 'gan1' \
                          --n_heads '3' \
                          --n_epochs '453' \
                          --interval '4'

# Use augmentation
python -m src.train.mdgan --exp_name 'baselinemheads_lossgan1_augmentation1' \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '453' \
                          --interval '4' \
                          --augmentation '1' \
                          --aug_times '2'

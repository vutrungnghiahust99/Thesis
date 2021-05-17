python -m src.train_v1 --exp_name 'gan1_baseline' \
                       --loss_name 'gan1' \
                       --augmentation '1' \
                       --aug_times '5' \
                       --dist 'gauss' \
                       --n_epochs '1000' \
                       --n_heads '1' \
                       --interval '8'

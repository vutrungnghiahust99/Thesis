python -m src.train_hgan --exp_name 'gan1_1head' \
                         --loss_name 'gan1' \
                         --augmentation '1' \
                         --aug_times '2' \
                         --dist 'gauss' \
                         --n_epochs '1000' \
                         --n_heads '1' \
                         --interval '2'
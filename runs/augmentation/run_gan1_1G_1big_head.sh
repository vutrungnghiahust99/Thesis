python -m src.train_v1 --exp_name 'gan1_1G_1big_head' \
                       --loss_name 'gan1' \
                       --augmentation '1' \
                       --aug_times '5' \
                       --dist 'gauss' \
                       --n_epochs '300' \
                       --n_heads '1' \
                       --interval '4' \
                       --use_big_head_d '1'

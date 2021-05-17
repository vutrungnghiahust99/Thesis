python -m src.train_v1 --exp_name 'lsgan_10_heads_1G_10heads' \
                       --loss_name 'lsgan' \
                       --augmentation '1' \
                       --aug_times '5' \
                       --dist 'gauss' \
                       --n_epochs '1000' \
                       --n_heads '10' \
                       --interval '8'

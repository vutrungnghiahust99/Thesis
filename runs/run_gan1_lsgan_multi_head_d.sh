python -m src.train_rf --exp_name 'gan1_multihead_d' \
                      --loss_name 'gan1' \
                      --augmentation '0' \
                      --spec_g '0' \
                      --spec_d '0' \
                      --dist 'gauss' \
                      --n_epochs '200' \
                      --interval '1'

python -m src.train_rf --exp_name 'lsgan_multihead_d' \
                      --loss_name 'lsgan' \
                      --augmentation '0' \
                      --spec_g '0' \
                      --spec_d '0' \
                      --dist 'gauss' \
                      --n_epochs '200' \
                      --interval '1'

python -m src.train --exp_name 'lsgan_10_heads_1G_10heads' \
                    --loss_name 'lsgan' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '1' \
                    --dist 'gauss' \
                    --n_epochs '200' \
                    --n_heads '10' \
                    --interval '2' \
                    --diff_data_for_heads '0'

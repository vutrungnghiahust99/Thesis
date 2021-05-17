python -m src.train --exp_name 'lsgan_base' \
                    --loss_name 'lsgan' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '1' \
                    --dist 'gauss' \
                    --n_epochs '300' \
                    --n_heads '1' \
                    --interval '2'

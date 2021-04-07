python -m src.train --exp_name 'base_gan1' \
                    --loss_name 'gan1' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '0' \
                    --dist 'gauss' \
                    --n_epochs '200' \
                    --interval '1'

python -m src.train --exp_name 'base_lsgan' \
                    --loss_name 'lsgan' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '0' \
                    --dist 'gauss' \
                    --n_epochs '200' \
                    --interval '1'

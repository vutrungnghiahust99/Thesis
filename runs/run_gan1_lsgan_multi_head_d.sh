python -m src.train --exp_name 'gan1_10_heads' \
                    --loss_name 'gan1' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '1' \
                    --dist 'gauss' \
                    --n_epochs '200' \
                    --n_heads '10' \
                    --interval '2'


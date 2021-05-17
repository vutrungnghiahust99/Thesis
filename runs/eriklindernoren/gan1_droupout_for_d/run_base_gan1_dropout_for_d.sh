python -m src.train --exp_name 'gan1_base_dropout_for_d' \
                    --loss_name 'gan1' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '1' \
                    --dist 'gauss' \
                    --n_epochs '300' \
                    --n_heads '1' \
                    --interval '2' \
                    --use_dropout '1'


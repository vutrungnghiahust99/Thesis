python -m src.train --exp_name 'gan1_1G_1big_head' \
                    --loss_name 'gan1' \
                    --augmentation '0' \
                    --spec_g '0' \
                    --spec_d '1' \
                    --dist 'gauss' \
                    --n_epochs '300' \
                    --n_heads '1' \
                    --interval '2' \
                    --diff_data_for_heads '0' \
                    --use_big_head_d '1'
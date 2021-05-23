# Do not use augmentation
python -m src.train.baseline_1head --exp_name 'baseline1head_lossgan1_augmentation0_usebighead0' \
                                   --loss_name 'gan1' \
                                   --use_big_head '0' \
                                   --n_epochs '401' \
                                   --interval '4'

python -m src.train.baseline_1head --exp_name 'baseline1head_lossgan1_augmentation1_usebighead0' \
                                   --loss_name 'gan1' \
                                   --use_big_head '1' \
                                   --n_epochs '401' \
                                   --interval '4'

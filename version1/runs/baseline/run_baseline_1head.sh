# Do not use augmentation
python -m src.train.baseline_1head --exp_name 'baseline1head_lossgan1_augmentation0_usebighead0' \
                                   --loss_name 'gan1' \
                                   --use_big_head '0' \
                                   --n_epochs '453' \
                                   --interval '4'

python -m src.train.baseline_1head --exp_name 'baseline1head_lossgan1_augmentation0_usebighead1' \
                                   --loss_name 'gan1' \
                                   --use_big_head '1' \
                                   --n_epochs '453' \
                                   --interval '4'

# Use augmentation
python -m src.train.baseline_1head --exp_name 'baseline1head_lossgan1_augmentation1_usebighead0' \
                                   --loss_name 'gan1' \
                                   --use_big_head '0' \
                                   --n_epochs '453' \
                                   --interval '4' \
                                   --augmentation '1' \
                                   --aug_times '2'

python -m src.train.baseline_1head --exp_name 'baseline1head_lossgan1_augmentation1_usebighead1' \
                                   --loss_name 'gan1' \
                                   --use_big_head '1' \
                                   --n_epochs '453' \
                                   --interval '4' \
                                   --augmentation '1' \
                                   --aug_times '2'

python -m src.train.rfgan --exp_name rfgan_lossgan1_bagging0_augmentation0_10heads \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '401' \
                          --interval '4'

python -m src.train.rfgan --exp_name rfgan_lossgan1_bagging1_augmentation0_10heads \
                          --loss_name 'gan1' \
                          --n_heads '10' \
                          --n_epochs '401' \
                          --interval '4' \
                          --bagging '1'

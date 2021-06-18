# Do not use bagging
python -m src.train.rfgan_eval_nDs --exp_name rfgan_lossgan1_bagging0_augmentation0_10heads_usemaskD_eval_nDs \
                                   --loss_name 'gan1' \
                                   --n_heads '10' \
                                   --n_epochs '201' \
                                   --interval '4' \
                                   --use_mask_d '1' \
                                   --data_path 'data/mnist/MNIST/processed/training_rf_10H.pt'

python -m src.train.rfgan_eval_nDs --exp_name rfgan_lossgan1_bagging0_augmentation0_50heads_usemaskD_eval_nDs \
                                   --loss_name 'gan1' \
                                   --n_heads '50' \
                                   --n_epochs '201' \
                                   --interval '4' \
                                   --use_mask_d '1' \
                                   --data_path 'data/mnist/MNIST/processed/training_rf_50H.pt'

python -m src.train.rfgan_eval_nDs --exp_name rfgan_lossgan1_bagging0_augmentation0_100heads_usemaskD_eval_nDs \
                                   --loss_name 'gan1' \
                                   --n_heads '100' \
                                   --n_epochs '201' \
                                   --interval '4' \
                                   --use_mask_d '1' \
                                   --data_path 'data/mnist/MNIST/processed/training_rf_100H.pt'


# use bagging
python -m src.train.rfgan_eval_nDs --exp_name rfgan_lossgan1_bagging1_augmentation0_10heads_usemaskD_eval_nDs \
                                   --loss_name 'gan1' \
                                   --n_heads '10' \
                                   --n_epochs '201' \
                                   --interval '4' \
                                   --bagging '1' \
                                   --use_mask_d '1' \
                                   --data_path 'data/mnist/MNIST/processed/training_rf_10H.pt'

python -m src.train.rfgan_eval_nDs --exp_name rfgan_lossgan1_bagging1_augmentation0_50heads_usemaskD_eval_nDs \
                                   --loss_name 'gan1' \
                                   --n_heads '50' \
                                   --n_epochs '201' \
                                   --interval '4' \
                                   --bagging '1' \
                                   --use_mask_d '1' \
                                   --data_path 'data/mnist/MNIST/processed/training_rf_50H.pt'

python -m src.train.rfgan_eval_nDs --exp_name rfgan_lossgan1_bagging1_augmentation0_100heads_usemaskD_eval_nDs \
                                   --loss_name 'gan1' \
                                   --n_heads '100' \
                                   --n_epochs '201' \
                                   --interval '4' \
                                   --bagging '1' \
                                   --use_mask_d '1' \
                                   --data_path 'data/mnist/MNIST/processed/training_rf_100H.pt'

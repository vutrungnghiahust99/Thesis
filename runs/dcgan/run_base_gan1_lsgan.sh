python -m src.train_dcgan --exp_name 'gan1_base' \
                   	   --loss_name 'gan1' \
                  	   --augmentation '0' \
                 	   --dist 'gauss' \
                  	   --n_epochs '300' \
                	   --n_heads '1' \
                 	   --interval '5'

python -m src.train_dcgan --exp_name 'lsgan_base' \
                          --loss_name 'lsgan' \
                 	   --augmentation '0' \
                          --dist 'gauss' \
                          --n_epochs '300' \
                          --n_heads '1' \
                          --interval '2'


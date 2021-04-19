python -m src.train_dcgan --exp_name 'gan1_4_heads_1G_4heads' \
	                  --loss_name 'gan1' \
		          --augmentation '0' \
		          --dist 'gauss' \
		          --n_epochs '300' \
		          --n_heads '4' \
		          --interval '5' \
		          --diff_data_for_heads '0'


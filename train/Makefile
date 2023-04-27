DATA_URL_GENERATE_BRIDGE=172.30.0.54:/volume1/数据使用/模型训练数据集/冠脉产品线/冠脉心肌桥分割模型/3.x/复盘版本/训练数据/processed_data
DATA_URL_ORIGINAL_BRIDGE=172.30.0.54:/volume1/数据使用/模型训练数据集/冠脉产品线/冠脉心肌桥分割模型/3.x/复盘版本/训练数据/origin_data
DATA_URL_PRETRAIN_WEIGHT=172.30.0.54:/volume1/数据使用/模型训练数据集/冠脉产品线/冠脉心肌桥分割模型/3.x/复盘版本/训练数据/pretrain_weight

DATA_DIR=train_data/
MODEL_ROOT=checkpoints

DATA_DIR_GENERATE_BRIDGE=train_data/processed_data
# make USER_NAME=..
run: download
	bash run_dist.sh -d 0,1,2,3 -g 4 -c ./config/seg_myocardialbridge_config.py

format:
	pre-commit run --all-files --show-diff-on-failure

tensorboard:
	tensorboard serve --logdir checkpoints/ --host '0.0.0.0'

download:
	test $(USER_NAME)
	rsync -avP $(USER_NAME)@$(DATA_URL_PRETRAIN_WEIGHT) train_data; \
	if [ ! -d "$(DATA_DIR)/processed_data" ]; then \
		echo "Downloading training data..."; \
		rsync -avP $(USER_NAME)@$(DATA_URL_GENERATE_BRIDGE) train_data; \
	fi

generate_data:
	test $(USER_NAME)
	rsync -avP $(USER_NAME)@$(DATA_URL_PRETRAIN_WEIGHT) train_data; \
	if [ ! -d "$(DATA_DIR)/origin_data"  ]; then \
		echo "Downloading original data..."; \
		rsync -avP $(USER_NAME)@$(DATA_URL_ORIGINAL_BRIDGE) train_data; \
	fi
	python3.7 custom/utils/generate_dataset.py --src_path ./train_data/origin_data --tgt_path  ./train_data/processed_data

save_torchscript:
	python3.7 custom/utils/save_torchscript.py --model_path ./checkpoints/v7/epoch_2.pth --config_path ./config/seg_myocardialbridge_config.py --output_path ./checkpoints

clean:
	rm -rf $(MODEL_ROOT)
	rm -rf $(DATA_DIR)

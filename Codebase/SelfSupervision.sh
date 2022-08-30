python run_t5_mlm_flax.py \
	--output_dir="../TrainedModels/PreTrainedT5" \
	--model_name_or_path="t5-base" \
	--train_file="../Datasets/Texts/train.txt" \
  --validation_file="../Datasets/Texts/validation.txt" \
  --use_auth_token=False \
	--max_seq_length="512" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500"
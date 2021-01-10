python local_train_mlm.py \
  --ln_type post \
  --output_dir no_fixup_post_big_acc32_nofp16 --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --learning_rate 0.0001 --max_grad_norm 25.0 \
  --max_steps 50000 --warmup_steps 5000 \
  --logging_dir logs/no_fixup_post_big_acc32_nofp16 --logging_first_step \
  --save_steps 5000 --save_total_limit 5 \
  --seed 0 --eval_steps 5000 \
  --dataloader_num_workers 4

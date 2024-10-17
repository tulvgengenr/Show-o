#! /bin/sh
accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml \
                  --main_process_port=8888 training/train_ti2ss.py \
                  config=configs/showo_instruction_tuning_2_ti2ss.yaml
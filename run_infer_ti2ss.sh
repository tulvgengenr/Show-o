#! /bin/sh
# python3 inference_ti2ss.py \
#     config=configs/showo_demo.yaml \
#     batch_size=1 \
#     validation_prompts_file=validation_prompts/ti2ss_demo.txt \
#     image_path=ti2ss_validation/bus.jpg  \
#     guidance_scale=0 \
#     generation_timesteps=18

python3 inference_ti2ss.py \
    config=configs/showo_demo.yaml \
    batch_size=1 \
    validation_prompts_file=validation_prompts/ti2ss_demo.txt \
    image_path=ti2ss_validation/bus.jpg  \
    guidance_scale=1.75 \
    generation_timesteps=18
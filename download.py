import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import huggingface_hub
huggingface_hub.login("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") 

from transformers import AutoTokenizer, AutoProcessor, Blip2ForConditionalGeneration, AutoModel
from models import Showo, MAGVITv2, get_mask_chedule


model_name_1 = "Salesforce/blip2-flan-t5-xl"
model_name_2 = "microsoft/phi-1_5"
model_name_3 = "showlab/magvitv2"
model_name_4 = "showlab/show-o-512x512-wo-llava-tuning"

# blip2_processor = AutoProcessor.from_pretrained(model_name_1)
# blip2_model = Blip2ForConditionalGeneration.from_pretrained(model_name_1)


# tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2, padding_side="left")



# def get_vq_model_class(model_type):
#     if model_type == "magvitv2":
#         return MAGVITv2
#     elif model_type == "vq16":
#         return VQ_16
#     else:
#         raise ValueError(f"model_type {model_type} not supported.")

# # VQ model for processing image into discrete tokens
# vq_model = get_vq_model_class("magvitv2")
# vq_model = vq_model.from_pretrained(model_name_3)



# Initialize Show-o model
model = Showo.from_pretrained(model_name_4)

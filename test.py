import random
from mmengine.registry import init_default_scope
from mmseg.structures import SegDataSample
from mmseg.datasets import ADE20KDataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch


init_default_scope('mmseg')

# 注册自定义数据集类
class CustomADE20KDataset(ADE20KDataset):
    def __init__(self, data_root, resolution=512):
        # Data arguments
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
        )
        crop_size = (resolution, resolution)

        ratio_range = (0.5, 2.0)
        ratio = random.uniform(ratio_range[0], ratio_range[1])
        if ratio < 1:
            scale_factor = (ratio, 1)
        else:
            scale_factor = (1, 1 / ratio)

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', 
                 scale=(2048, 512), 
                 scale_factor=scale_factor,
                 keep_ratio=False,
            ),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=dict(img=0, seg=255)),
            dict(type='PackSegInputs')
        ]

        super().__init__(
            data_root = data_root,
            data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            pipeline=train_pipeline,
        )
        
        # BLIP-2
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float32)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        image_ts = data['inputs']
        data_sample:SegDataSample = data['data_samples']
        seg_map_ts = data_sample.gt_sem_seg.data
        


        return dict(image=image_ts, seg_map=seg_map_ts, data_samples=data_sample)


# Example usage:
training_dataset = CustomADE20KDataset(data_root='/data/junzhe/datasets/ade/ADEChallengeData2016', resolution=512)
print(training_dataset.get_data_info(0))
print(training_dataset.metainfo)

seg_map = training_dataset[0]['seg_map']



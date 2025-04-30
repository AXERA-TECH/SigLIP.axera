import numpy as np
import torch
import numpy as np
import logging
import time
import torch
import os
from tqdm import tqdm
from model_convert.cali_data.imagenet_dataset import ImagenetDataset, imagenet_classes, imagenet_templates
import onnx
import onnxruntime as ort
import numpy as np
import random
from transformers import AutoProcessor, AutoModel
import torch

# model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")


for i, classname in enumerate(imagenet_classes):
    if i>=64:
        break

    idx = random.randint(0, 79)

    texts = [imagenet_templates[idx].format(classname)]
    # format with class
    inputs = processor(text=texts, images=None, padding="max_length", return_tensors="pt")
    texts = inputs.data['input_ids']
    s_path = f"/data/baizanzhou/project/inner/siglip/huggingface/cali_data/text_cali/{idx}.npy"
    print("save: ", s_path, texts.shape)
    np.save(s_path, texts.numpy())

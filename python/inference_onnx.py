from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import onnxruntime as ort
import numpy as np
import torch.nn as nn

def model_inference(model_path,inputs):
    vision_sess = ort.InferenceSession(model_path)
    input_name = vision_sess.get_inputs()[0].name
    outputs = [vision_sess.run(None, {input_name: np.array([i])})[1] for i in inputs]
    outputs = np.array(outputs).reshape(-1,1152)
    return outputs

# get input data
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

# inference onnx
vision_outputs = model_inference("/data/baizanzhou/project/inner/siglip/huggingface/model_convert/onnx/siglip-so400m-patch14-384_vision.onnx",inputs['pixel_values'])
text_outputs = model_inference("/data/baizanzhou/project/inner/siglip/huggingface/model_convert/onnx/siglip-so400m-patch14-384_text.onnx",inputs['input_ids'])

image_embeds = torch.tensor(vision_outputs)
text_embeds = torch.tensor(text_outputs)

# normalized features
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# cosine similarity as logits
logit_scale = np.random.randn(1)[0]#nn.Parameter(torch.randn(1))
logit_bias = np.random.randn(1)[0]# nn.Parameter(torch.randn(1))
logits_per_text = (
    np.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * np.exp(logit_scale)
    + logit_bias
)
logits_per_image = logits_per_text.t()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
probs = sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
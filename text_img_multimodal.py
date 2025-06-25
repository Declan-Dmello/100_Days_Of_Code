import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("Declan.png")
texts = [ "A bus", "A traffic cone" , "Dog"]
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item()*100:.2f}%")
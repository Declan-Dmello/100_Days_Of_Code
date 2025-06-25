import torch
import timm
import torchvision.transforms as transforms
from PIL import Image


class VisionTransformerInference:
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1000
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):

        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_classes = torch.topk(probs, 5)

        return top5_prob, top5_classes


def main():
    vit = VisionTransformerInference()
    image_path = 'Screenshot 2024-06-23 154303.png'

    top5_prob, top5_classes = vit.predict(image_path)

    print("Top 5 Predictions:")
    for prob, cls in zip(top5_prob[0], top5_classes[0]):
        print(f"Class {cls.item()}: {prob.item() * 100:.2f}%")


if __name__ == '__main__':
    main()
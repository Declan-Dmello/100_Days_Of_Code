import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, ViTModel, ViTFeatureExtractor
import torch.nn.functional as F

class MultiModalTransformer(nn.Module):
    def __init__(self, text_model_name, vision_model_name, tabular_input_size, num_classes):
        super(MultiModalTransformer, self).__init__()

        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_embedding_size = self.text_model.config.hidden_size

        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        self.vision_feature_extractor = ViTFeatureExtractor.from_pretrained(vision_model_name)
        self.vision_embedding_size = self.vision_model.config.hidden_size


        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        self.tabular_embedding_size = 64


        self.fc = nn.Sequential(
            nn.Linear(self.text_embedding_size + self.vision_embedding_size + self.tabular_embedding_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, text_inputs, image_inputs, tabular_inputs):

        text_tokens = self.text_tokenizer(
            text_inputs, padding=True, truncation=True, return_tensors="pt"
        )
        text_embeddings = self.text_model(**text_tokens).last_hidden_state[:, 0, :]

        image_features = self.vision_feature_extractor(
            images=image_inputs, return_tensors="pt"
        )
        image_embeddings = self.vision_model(**image_features).last_hidden_state[:, 0, :]


        tabular_embeddings = self.tabular_fc(tabular_inputs)

        combined_embeddings = torch.cat([text_embeddings, image_embeddings, tabular_embeddings], dim=1)
        outputs = self.fc(combined_embeddings)
        return outputs


text_model_name = "bert-base-uncased"
vision_model_name = "google/vit-base-patch16-224"
tabular_input_size = 10  # Number of tabular features
num_classes = 10  # Adjust based on your task


model = MultiModalTransformer(text_model_name, vision_model_name, tabular_input_size, num_classes)


dummy_text = ["A sunny day by the beach", "A cat meowing in a quiet room"]
dummy_images = [torch.rand(3, 224, 224), torch.rand(3, 224, 224)]  # Replace with actual images
dummy_tabular = torch.rand(2, tabular_input_size)  # Replace with real tabular data (batch_size x features)

outputs = model(dummy_text, dummy_images, dummy_tabular)


probs = F.softmax(outputs, dim=1)
print(probs)
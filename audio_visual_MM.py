import torch
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model


class AudioVisualFusionModel:
    def __init__(self):
        self.video_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.video_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

        self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def preprocess_video(self, video_tensor):
        video_images = []
        for v in video_tensor:
            img_array = ((v - v.min()) / (v.max() - v.min()) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            video_images.append(Image.fromarray(img_array))


        video_inputs = self.video_extractor(video_images, return_tensors="pt")
        return video_inputs

    def preprocess_audio(self, audio_tensor):
        audio_np = audio_tensor.numpy()
        audio_inputs = self.audio_extractor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt"
        )
        return audio_inputs

    def fuse_features(self, video_tensor, audio_tensor):
        video_inputs = self.preprocess_video(video_tensor)
        video_outputs = self.video_model(**video_inputs)
        video_features = video_outputs.logits

        audio_inputs = self.preprocess_audio(audio_tensor)
        audio_outputs = self.audio_model(**audio_inputs)
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)

        if video_features.size(1) != audio_features.size(1):
            feature_dim = min(video_features.size(1), audio_features.size(1))
            video_features = video_features[:, :feature_dim]
            audio_features = audio_features[:, :feature_dim]

        fused_features = torch.cat([video_features, audio_features], dim=1)

        return fused_features


def prepare_data():
    video = torch.rand(4, 3, 224, 224)
    audio = torch.randn(4, 16000)
    return video, audio


model = AudioVisualFusionModel()
video, audio = prepare_data()
result = model.fuse_features(video, audio)
print("Fused Features Shape:", result.shape)

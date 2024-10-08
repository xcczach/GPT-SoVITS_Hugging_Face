
import torch
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    HubertConfig,
    AutoConfig
)

import torch.nn as nn

class CNHubert(nn.Module):
    def __init__(self, hubert_config_dict: dict[str, any], extractor_config_dict: dict[str, any]):
        super().__init__()
        self.model = HubertModel(HubertConfig.from_dict(hubert_config_dict))
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_dict(extractor_config_dict)

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats


# class CNHubertLarge(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = HubertModel.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-hubert-large")
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-hubert-large")
#     def forward(self, x):
#         input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
#         feats = self.model(input_values)["last_hidden_state"]
#         return feats
#
# class CVec(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = HubertModel.from_pretrained("/data/docker/liujing04/vc-webui-big/hubert_base")
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/docker/liujing04/vc-webui-big/hubert_base")
#     def forward(self, x):
#         input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
#         feats = self.model(input_values)["last_hidden_state"]
#         return feats
#
# class cnw2v2base(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = Wav2Vec2Model.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-wav2vec2-base")
#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/docker/liujing04/gpt-vits/chinese-wav2vec2-base")
#     def forward(self, x):
#         input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
#         feats = self.model(input_values)["last_hidden_state"]
#         return feats


# def get_large_model():
#     model = CNHubertLarge()
#     model.eval()
#     return model
#
# def get_model_cvec():
#     model = CVec()
#     model.eval()
#     return model
#
# def get_model_cnw2v2base():
#     model = cnw2v2base()
#     model.eval()
#     return model


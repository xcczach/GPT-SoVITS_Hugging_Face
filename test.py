from gpt_sovits_model.configuration_gpt_sovits import GPTSoVITSConfig
from gpt_sovits_model.modeling_gpt_sovits import GPTSoVITSModel
from transformers import HubertModel, Wav2Vec2FeatureExtractor

GPTSoVITSConfig.register_for_auto_class()
GPTSoVITSModel.register_for_auto_class("AutoModel")

hubert_model = HubertModel.from_pretrained("./ckpts/chinese-hubert-base")
extractor = Wav2Vec2FeatureExtractor.from_pretrained("./ckpts/chinese-hubert-base")
model_config = GPTSoVITSConfig(prompt_language="zh",device="cpu",is_half=True,
                               _hubert_config_dict=hubert_model.config.to_dict(),
                               _hubert_extractor_config_dict=extractor.to_dict(),
)
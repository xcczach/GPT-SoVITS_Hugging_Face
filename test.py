from yags_model.configuration_yags import GPTSoVITSConfig
from yags_model.modeling_yags import GPTSoVITSModel
from transformers import HubertModel, Wav2Vec2FeatureExtractor, AutoModelForMaskedLM, AutoTokenizer
import torch
from contextlib import contextmanager
import os
import shutil
import soundfile as sf
from utils import HParams

def hparams_to_dict(hparams: HParams):
    result = {}
    for key, value in hparams.items():
        if isinstance(value, HParams):
            result[key] = hparams_to_dict(value)
        else:
            result[key] = value
    return result

@contextmanager
def load_files_temp():
    target_dir = "yags_model"
    to_upload_dir = "to_upload"
    for file in os.listdir(to_upload_dir):
        shutil.copy(os.path.join(to_upload_dir, file), os.path.join(target_dir, file))
    yield
    for file in os.listdir(to_upload_dir):
        os.remove(os.path.join(target_dir, file))

with load_files_temp():
    GPTSoVITSConfig.register_for_auto_class()
    GPTSoVITSModel.register_for_auto_class("AutoModel")

    hubert_model = HubertModel.from_pretrained("./ckpts/chinese-hubert-base")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("./ckpts/chinese-hubert-base")
    bert_path = "./ckpts/chinese-roberta-wwm-ext-large"
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    dict_s1 = torch.load("./ckpts/tts_model/sunshine_girl.ckpt", map_location="cpu")
    dict_s2 = torch.load("./ckpts/tts_model/sunshine_girl.pth", map_location="cpu")
    hps_dict = hparams_to_dict(dict_s2["config"])
    model_config = GPTSoVITSConfig(prompt_language="zh",
                                _hubert_config_dict=hubert_model.config.to_dict(),
                                _hubert_extractor_config_dict=extractor.to_dict(),
                                _bert_config_dict=bert_model.config.to_dict(),
                                _hps_dict=hps_dict,
                                _gpt_config_dict=dict_s1["config"],
    )
    model = GPTSoVITSModel(model_config)
    model.bert_model.load_state_dict(bert_model.state_dict())
    model.ssl_model.model.load_state_dict(hubert_model.state_dict())
    model.vq_model.load_state_dict(dict_s2["weight"], strict=False)
    model.t2s_model.load_state_dict(dict_s1["weight"])
    model.to("cuda").half()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    speech_arr, sr = model.infer("我真的不是一袋猫粮呢！",tokenizer=tokenizer)
    sf.write("output.wav", speech_arr, sr)
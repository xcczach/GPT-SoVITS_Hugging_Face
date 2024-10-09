from transformers import PretrainedConfig
import torch


class GPTSoVITSConfig(PretrainedConfig):
    model_type = "gpt_sovits"

    def __init__(
            self,
            prompt_language: str,
            _hubert_config_dict: dict[str, any] = None,
            _hubert_extractor_config_dict: dict[str, any] = None,
            _bert_config_dict: dict[str, any] = None,
            _hps_dict: dict[str, any] = None,
            _gpt_config_dict: dict[str, any] = None,
            **kwargs
    ):
        self.prompt_language = prompt_language
        self._hubert_config_dict = _hubert_config_dict
        self._hubert_extractor_config_dict = _hubert_extractor_config_dict
        self._bert_config_dict = _bert_config_dict
        self._hps_dict = _hps_dict
        self._gpt_config_dict = _gpt_config_dict

        super().__init__(**kwargs)
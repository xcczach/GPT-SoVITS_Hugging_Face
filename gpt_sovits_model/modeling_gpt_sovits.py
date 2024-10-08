from transformers import PreTrainedModel
from .configuration_gpt_sovits import GPTSoVITSConfig

class GPTSoVITSModel(PreTrainedModel):
    config_class = GPTSoVITSConfig

    def __init__(self, config: GPTSoVITSConfig):
        super().__init__(config)
        self.prompt_language = config.prompt_language

    def forward(self, input_ids, **kwargs):
        return input_ids
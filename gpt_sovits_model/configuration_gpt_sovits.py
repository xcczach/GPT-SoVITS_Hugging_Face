from transformers import PretrainedConfig
import torch
torch.device()

class GPTSoVITSConfig(PretrainedConfig):
    model_type = "gpt_sovits"

    def __init__(
            self,
            prompt_language: str,
            is_half: bool = True,
            device: str | int | torch.device = "cpu",
            **kwargs
    ):
        self.prompt_language = prompt_language
        self.is_half = is_half
        self.device = device
        
        super().__init__(**kwargs)
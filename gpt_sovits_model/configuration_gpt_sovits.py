from transformers import PretrainedConfig

class GPTSoVITSConfig(PretrainedConfig):
    model_type = "gpt_sovits"

    def __init__(
            self,
            prompt_language: str,
            **kwargs
    ):
        self.prompt_language = prompt_language
        super().__init__(**kwargs)
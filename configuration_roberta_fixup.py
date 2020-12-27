from transformers.models.roberta.configuration_roberta import RobertaConfig


class RobertaFixupConfig(RobertaConfig):
    model_type = 'roberta_fixup'

    def __init__(self, use_fixup=False, ln_type='post', **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(**kwargs)
        self.use_fixup = use_fixup
        self.ln_type = ln_type

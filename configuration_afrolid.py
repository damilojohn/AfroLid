from transformers import PretrainedConfig
from dataclasses import dataclass, asdict

# @dataclass
# class QuantNoiseConfig:
#     pq: float = 0.0
#     pq_block_size: int = 8

@dataclass
class QuantNoiseConfig:
    pq: float = 0.0
    pq_block_size: int = 8

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


repo_name = "damilojohn/AfroLid"


class AfroLidConfig(PretrainedConfig):
    model_type = "afrolid"

    def __init__(self,
                 encoder_vocab_size=64001,
                 decoder_vocab_size=528,
                 embed_dim=768,
                 ffn_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 max_seq_len=512,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation_dropout=0.0,
                 layerdrop=0.0,
                 normalize_before=False,
                 learned_pos=False,
                 max_source_positions=1024,
                 max_target_positions=1024,
                 no_token_positional_embeddings=False,
                 share_decoder_input_output_embed=True,
                 share_all_embeddings=False,
                 layernorm_embedding=False,
                 checkpoint_activations=False,
                 offload_activations=False,
                 bias=False,
                 **kwargs):
        """
        AfroLid configuration class for an encoder-decoder transformer model,
        with support for QuantNoiseConfig.
        """
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.normalize_before = normalize_before
        self.learned_pos = learned_pos
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.share_all_embeddings = share_all_embeddings
        self.layernorm_embedding = layernorm_embedding
        self.checkpoint_activations = checkpoint_activations
        self.offload_activations = offload_activations
        self.bias = bias

        super().__init__(**kwargs)

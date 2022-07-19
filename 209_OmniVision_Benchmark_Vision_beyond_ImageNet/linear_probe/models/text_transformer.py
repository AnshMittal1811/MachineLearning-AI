from collections import OrderedDict
from typing import Union, List
import logging
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

LAYER_NORM = True

__all__ = ['clip_text']


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if LAYER_NORM:
            ret = super().forward(x)
        else:
            ret = x
        return ret


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpoint: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.checkpoint = checkpoint
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input = input.detach()
            input.requires_grad = True
        return checkpoint_sequential(layer, segments, input)

    def forward(self, x: torch.Tensor):
        if self.checkpoint:
            return self.checkpoint_fwd(self.resblocks, x, self.layers)
        return self.resblocks(x)


class CLIP_text(nn.Module):
    def __init__(
        self, 
        positional_embedding_flag=True, 
        transformer_width=512,
        context_length=77,
        vocab_size=49408,
        embed_dim=1024,
        transformer_layers=12,
        transformer_heads=8,
        layer_norm=True,
        text_projection_with_bias=True,
    ):
        super().__init__()

        self.context_length = context_length
        self.positional_embedding_flag = positional_embedding_flag
        global LAYER_NORM
        LAYER_NORM = layer_norm

        self.tokenizer = _Tokenizer()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.context_length, transformer_width)))  # Fix!!!
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection_with_bias = text_projection_with_bias
        if text_projection_with_bias:
            self.text_projection = nn.Linear(transformer_width, embed_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # text transformer init
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if not self.text_projection_with_bias:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]

        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length-1] + [tokens[-1]]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def encode_text(self, texts):
        texts = self.tokenize(texts, context_length=self.context_length)
        x = self.token_embedding(texts.cuda()).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if self.positional_embedding_flag:
            x = x + self.positional_embedding.type(self.dtype)  # Fix!!!
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)]
        if self.text_projection_with_bias:
            x = self.text_projection(x)
        else:
            x = x @ self.text_projection

        return x

    def forward(self, texts):
        return self.encode_text(texts)


def load_clip_state_text_model(model, ckpt_path):
    ckpt_state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt_state:
        # our gvm-clip checkpoint
        ckpt_state = ckpt_state['state_dict']
        prefix = 'module.text_model'
        exclude_prefix = None
    elif 'model' in ckpt_state:
        # prototype checkpoint
        ckpt_state = ckpt_state['model']
        prefix = 'module.'
        exclude_prefix = 'module.visual.'
    else:
        # OpenAI checkpoint
        prefix = ''
        exclude_prefix = 'visual.'

    logger = logging.getLogger('global')
    if ckpt_state:
        logger.info('==> Loading text model state "{}XXX" from CLIP model..'.format(prefix))
        
        own_state = model.state_dict()
        state = {}
        for name, param in ckpt_state.items():
            if name.startswith(prefix):
                if exclude_prefix is not None and name.startswith(exclude_prefix):
                    continue
                state[name[len(prefix):]] = param
        success_cnt = 0
        for name, param in state.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    if isinstance(param, bool):
                        own_state[name] = param
                    else:
                        # normal version 
                        own_state[name].copy_(param)
                    success_cnt += 1
                except Exception as err:
                    logger.warn(err)
                    logger.warn('while copying the parameter named {}, '
                                         'whose dimensions in the model are {} and '
                                         'whose dimensions in the checkpoint are {}.'
                                         .format(name, own_state[name].size(), param.size()))
                    logger.warn("But don't worry about it. Continue pretraining.")
        ckpt_keys = set(state.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        logger.info('Successfully loaded {} key(s) from {}'.format(success_cnt, ckpt_path))
        for k in missing_keys:
            logger.warn('Caution: missing key from text model of CLIP checkpoint: {}'.format(k))
        redundancy_keys = ckpt_keys - own_keys
        for k in redundancy_keys:
            logger.warn('Caution: redundant key from text model of CLIP checkpoint: {}'.format(k))


def clip_text(clip_pretrain_path=None, **kwargs):
    model = CLIP_text(**kwargs)
    if clip_pretrain_path is not None:
        load_clip_state_text_model(model, clip_pretrain_path)
    return model

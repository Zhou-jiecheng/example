import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .activations import GELU
from torch.cuda.amp import autocast
from .linear_fuc import get_linear_cls
from typing import Optional, Tuple, Union


class GPT2Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

        
    def forward(self, input_ids: Optional[torch.LongTensor]):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        self.position_ids = torch.arange(0, self.max_position_embeddings, dtype=torch.long).unsqueeze(0).view(-1, self.max_position_embeddings).cuda()
        #print("position_ids device:",self.position_ids.device())
        position_embeds = self.wpe(self.position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states
#linear
class Conv1D(nn.Module):
    def __init__(self, out_features, in_features):
        #super().__init__()
        super().__init__()
        self.nf = out_features
        #self.weight = nn.Parameter(torch.empty(nf, nx))
        #self.bias = nn.Parameter(torch.zeros(nf))
        #nn.init.normal_(self.weight, std=0.02)
        self.linear = get_linear_cls()(in_features, out_features, True)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        #x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = self.linear(x.view(-1, x.size(-1)))
        x = x.view(size_out)
        return x

class GPT2MLP(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = Conv1D(in_dim, hidden_size)
        self.c_proj = Conv1D(hidden_size, in_dim)
        self.act = GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.is_cross_attention = config.is_cross_attention
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.causal=config.use_causal_mask

        self.pruned_heads = set()

    def _attn(self, query, key, value):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if not self.is_cross_attention and self.causal:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def _upcast_and_reordered_attn(self, query, key, value):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention and self.causal:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def orig_forward(
        self,
        hidden_states: Optional[torch.FloatTensor]
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if self.reorder_and_upcast_attn:
            attn_output = self._upcast_and_reordered_attn(query, key, value)
        else:
            attn_output = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output 
    
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor]
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        #return checkpoint(self.orig_forward, hidden_states)
        return self.orig_forward(hidden_states)
    
    
class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.use_flash_attention:
            self.attn = GPT2FlashAttention(config)
        else:
            self.attn = GPT2Attention(config)
        

        self.mlp = GPT2MLP(inner_dim, config)
        #self.mlp = FusedMLP(hidden_size, inner_dim, config.activation_function, config.resid_pdrop, use_torchscript=True)
    
    def forward_step1(
        self,
        hidden_states: Optional[torch.FloatTensor]
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = attn_output + residual
        return hidden_states
    
    def forward_step2(
        self,
        hidden_states: Optional[torch.FloatTensor] 
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor]
    ) -> torch.FloatTensor:
        #hidden_states = checkpoint(self.forward_step1, hidden_states)
        #hidden_states = checkpoint(self.forward_step2, hidden_states)
        hidden_states = self.forward_step2(self.forward_step1(hidden_states))
        return hidden_states
    
class GPT2Tails(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = get_linear_cls()(config.hidden_size, config.vocab_size, False)

    def forward(self, hidden_states: Optional[torch.FloatTensor]):
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states.view(-1, hidden_states.size(-1)))
        return hidden_states


import torch
from llama.model import LLaMA

seq_len = 10

context_len = 20
d_model = 256
n_heads = 8
n_blocks = 4
vocab_size = 10000
pos_emb_dropout_p = 0.1
pos_emb_type = "rope"
learned = False
ntk_rope_scaling = False
dyn_scaling = False
attn_type = "gqa"
n_groups = 4
supress_warnings = True

model = LLaMA(
    context_len = context_len,
    d_model=d_model,
    n_heads = n_heads,
    n_blocks = n_blocks,
    vocab_size = vocab_size,
    pos_emb_dropout_p = pos_emb_dropout_p,
    pos_emb_type = pos_emb_type,
    learned = learned,
    ntk_rope_scaling = ntk_rope_scaling,
    dyn_scaling = dyn_scaling,
    attn_type = attn_type,
    n_groups = n_groups,
    supress_warnings = supress_warnings
    )

x = torch.randint(low = 0, high = vocab_size, size = (2, seq_len))

print(model(x).shape) # 2, 10, 10000
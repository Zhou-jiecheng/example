from model.utils import PipeSequential
import model.GPT as gpt
import torch
import torch.nn as nn
import deepspeed

config = {
    "seq_length": 512,
    "n_layer": 22,
    "vocab_size": 50257,
    "hidden_size": 1024,
    "max_position_embeddings": 512,
    "embd_pdrop": 0.1,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "num_attention_heads": 16,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-05,
    "use_flash_attention": False,
    "use_causal_mask": True,
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict(config)
rank = torch.cuda.current_device()
torch.cuda.set_device(rank)

model = PipeSequential()
model.add_module("embedding", gpt.GPT2Embeddings(config))
for i in range(config.n_layer):
    model.add_module(f"block_{i}", gpt.GPT2Block(config))
model.add_module("tail", gpt.GPT2Tails(config))
model = model.cuda()
model_engine,opt,_,_=deepspeed.initialize(config="./config.json",
                                          model=model,
                                          model_parameters=[p for p in model.parameters()])

#print(model)
dummy_input = torch.randint(0, 50257,(10,512)).cuda()
output = model_engine.forward(dummy_input)
print(output.shape)
label = torch.randint(1,50000,(5120,50257)).cuda()
label = label.float()
loss_func = nn.CrossEntropyLoss()
loss = loss_func(output,label)
print("loss:",loss)
#print(output)
model_engine.backward(loss)
model_engine.step()
import torch
import deepspeed
import torch.nn as nn
from model.T5 import T5,DecoderLayer,EncoderLayer
from model.utils import PipeSequential
import model.T5 as t5
import torch.multiprocessing as mp
import os

torch.manual_seed(666)
T5_config = {
    "seq_length":1024,
    "n_layer":4,
    "vocab_size": 50257,
    "dec_vocab_size":50257,
    "hidden_size": 512,
    "max_position_embeddings": 1024,
    "embd_pdrop": 0.1,
    "num_attention_heads": 16,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "src_pad_idx":1,
    "trg_pad_idx":2,
    "trg_sos_idx":3
}

data_config = {
    "data_path": ["1", "./tmp/data/my-gpt2_text_document"],
    "data_impl": "mmap",
    "split": "949,50,1",
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict(T5_config)
data_config = dotdict(data_config)


'''
import os
master_addr = os.environ['MASTER_ADDR']
master_port = int(os.environ['MASTER_PORT'])
rank = torch.cuda.current_device()
deepspeed.init_distributed(dist_backend="nccl",init_method="env://",world_size=4,rank=rank)
torch.cuda.set_device()
t5_test = T5(config).cuda()

model_engine,opt,_,_=deepspeed.initialize(config="./config.json",
                                          model=t5_test,
                                          model_parameters=[p for p in t5_test.parameters()])

#print(model_engine)
print("rank:",rank)
dummy_input = torch.randint(1,50000,(10,1024)).cuda()
output = model_engine.forward(dummy_input,dummy_input)
#print(output)
#print(output.shape)
label = torch.randint(1,50000,(10,1024,50257)).cuda()
label = label.float()
loss_func = nn.CrossEntropyLoss()
loss = loss_func(output,label)
print("loss:",loss)
model_engine.backward(loss)
model_engine.step()


print(t5_test)
dummy_input = torch.randint(1,50000,(10,1024)).cuda()
output = t5_test(dummy_input,dummy_input)
print(output)

label = torch.randint(1,50000,(10,1024,50257)).cuda()
label = label.float()
output = t5_test(dummy_input,dummy_input)
print(output.shape)
loss_func = nn.CrossEntropyLoss()
loss = loss_func(output,label)
loss.backward()
'''

dp_size=4
rank = torch.cuda.current_device()
#deepspeed.init_distributed(dist_backend="nccl",world_size=dp_size,rank=rank)
print(rank)

torch.cuda.set_device(rank)
t5_test = T5(config).cuda()
model_engine,opt,_,_=deepspeed.initialize(config="./config.json",
                                        model=t5_test,
                                        model_parameters=[p for p in t5_test.parameters()])

#print(model_engine)
print("rank:",rank)
dummy_input = torch.randint(1,50000,(10,1024)).cuda()
output = model_engine.forward(dummy_input,dummy_input)
#print(output)
#print(output.shape)
label = torch.randint(1,50000,(10,1024,50257)).cuda()
label = label.float()
loss_func = nn.CrossEntropyLoss()
loss = loss_func(output,label)
print("loss:",loss)
model_engine.backward(loss)
model_engine.step()

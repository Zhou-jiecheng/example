import torch
import torch.nn as nn
from model.bert import BERTLM,BERT
import deepspeed

bert_config = {
    "seq_length":1024,
    "n_layer":6,
    "vocab_size": 5000,
    "hidden_size": 1024,
    "max_position_embeddings": 1024,
    "pdrop": 0.1,
    "num_attention_heads": 32,
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict(bert_config)

#模型定义
Bert = BERT(config).cuda()
bertlm = BERTLM(Bert,5000).cuda()
model_engine,opt,_,_=deepspeed.initialize(config="./config.json",
                                        model=bertlm,
                                        model_parameters=[p for p in bertlm.parameters()])


dummy_input = torch.randint(1,5000,(1,1024)).cuda()
dummy_seg_label = torch.randint(1,2,(1,1024)).cuda()

a,b = model_engine.forward(dummy_input,dummy_seg_label)
loss_fuc = nn.CrossEntropyLoss()
label = torch.randint(1,5000,(1,1024,5000)).cuda()
label = label.float()

print(a.shape)
print(b.shape)

loss = loss_fuc(b,label)
print("loss:",loss)
model_engine.backward(loss)
model_engine.step()
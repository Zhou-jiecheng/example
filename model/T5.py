import torch
import torch.nn as nn
import math
from torch.nn import LayerNorm


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len).cuda()
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float().cuda()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))).cuda()
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))).cuda()
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, config):
        """
        class for word embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.pos_emb = PositionalEncoding(config.hidden_size, config.max_position_embeddings)
        self.drop_out = nn.Dropout(p=config.embd_pdrop)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        print(tok_emb.device)
        pos_emb = self.pos_emb(x).cuda()
        print(pos_emb.device)
        return self.drop_out(tok_emb + pos_emb)
'''
device = torch.cuda.current_device()
embedding = TransformerEmbedding(50247,1024,512,0.3).cuda()
dummy_input = torch.randint(1,50000,(10,1024)).cuda()
embedding_output = embedding(dummy_input)
print(embedding_output.shape)
'''

class ScaleDotProductAttention(nn.Module):
    def __init__(self, ):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-1, -2) # 计算矩阵 K 的转置  
        d_k = Q.size(-1)
        sequence_len = Q.size(-2)
        # 1, 计算 Q, K^T 矩阵的点积，再除以 sqrt(d_k) 得到注意力分数矩阵
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        # 2, 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
        if mask is not None:
            value_type = Q.dtype
            value = torch.finfo(value_type).min
            scores = scores.masked_fill(mask == 0, value)
        # 3, 对注意力分数矩阵按照最后一个维度进行 softmax 操作，得到注意力权重矩阵，值范围为 [0, 1]
        attn_weights = self.softmax(scores)
        # 4, 将注意力权重矩阵乘以 V，得到最终的输出矩阵
        output = torch.matmul(attn_weights, V)

        return output, attn_weights



class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer
    Args:
        d_model: Dimensions of the input embedding vector, equal to input and output dimensions of each head
        n_head: number of heads, which is also the number of parallel attention layers
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)  # Q 线性变换层
        self.w_k = nn.Linear(d_model, d_model)  # K 线性变换层
        self.w_v = nn.Linear(d_model, d_model)  # V 线性变换层
        self.fc = nn.Linear(d_model, d_model)   # 输出线性变换层

    def forward(self, q,k,v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # size is [batch_size, seq_len, d_model]
        # 2, split by number of heads(n_head) # size is [batch_size, n_head, seq_len, d_model//n_head]
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3, compute attention
        sa_output, attn_weights = self.attention(q, k, v, mask)
        # 4, concat attention and linear transformation
        concat_tensor = self.concat(sa_output)
        mha_output = self.fc(concat_tensor)

        return mha_output

    def split(self, tensor):
        """
        split tensor by number of head(n_head)

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, n_head, seq_len, d_model//n_head], 输出矩阵是四维的，第二个维度是 head 维度

        # 将 Q、K、V 通过 reshape 函数拆分为 n_head 个头
        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, n_head, d_model // n_head)
        k = k.reshape(batch_size, seq_len, n_head, d_model // n_head)
        v = v.reshape(batch_size, seq_len, n_head, d_model // n_head)
        """

        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        split_tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return split_tensor

    def concat(self, sa_output):
        """ merge multiple heads back together

        Args:
            sa_output: [batch_size, n_head, seq_len, d_tensor]
            return: [batch_size, seq_len, d_model]
        """
        batch_size, n_head, seq_len, d_tensor = sa_output.size()
        d_model = n_head * d_tensor
        concat_tensor = sa_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return concat_tensor

'''
Multihead = MultiHeadAttention(512,16)
Multihead = Multihead.cuda()
print(Multihead)
dummy_input = torch.randn(10,1024,512,dtype=torch.float32).cuda()#batch_size,sequence_length,hidden_size
Multihead_output = Multihead(dummy_input)
print(Multihead_output.shape)
'''

class FeedForward(nn.Module):
    def __init__(self, d_model, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
'''
FFN = FeedForward(512)
FFN = FFN.cuda()
dummy_input = torch.randn(10,1024,512,dtype=torch.float32).cuda()#batch_size,sequence_length,hidden_size
FFN_out = FFN(dummy_input)
print(FFN_out)
print(FFN_out.shape)
'''
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(config.hidden_size, config.num_attention_heads)
        self.ffn = FeedForward(config.hidden_size)
        self.ln1 = LayerNorm(config.hidden_size)
        self.ln2 = LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.attn_pdrop)
        self.dropout2 = nn.Dropout(config.attn_pdrop)

    def forward(self, x, mask=None):
        x_residual1 = x

        # 1, compute multi-head attention
        x = self.mha(x,x,x, mask=mask)

        # 2, add residual connection and apply layer norm
        x = self.ln1( x_residual1 + self.dropout1(x) )
        x_residual2 = x

        # 3, compute position-wise feed forward
        x = self.ffn(x)

        # 4, add residual connection and apply layer norm
        x = self.ln2( x_residual2 + self.dropout2(x) )

        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder,self).__init__()
        self.encoder_emb = TransformerEmbedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) 
                                     for _ in range(config.n_layer)])

    def forward(self, x, mask=None):

        x = self.encoder_emb(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x
'''
encoder = Encoder(50247,1024,512,16,6).cuda()
dummy_input = torch.randint(1,50000,(10,1024)).cuda()
encoder_out = encoder(dummy_input)
print(encoder_out)
print(encoder_out.shape)
'''

class DecoderLayer(nn.Module):

    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.config = config
        self.mha1 = MultiHeadAttention(config.hidden_size, config.num_attention_heads)
        self.ln1 = LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(p=config.attn_pdrop)

        self.mha2 = MultiHeadAttention(config.hidden_size, config.num_attention_heads)
        self.ln2 = LayerNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(p=config.attn_pdrop)

        self.ffn = FeedForward(config.hidden_size)
        self.ln3 = LayerNorm(config.hidden_size)
        self.dropout3 = nn.Dropout(p=config.attn_pdrop)

    def forward(self, dec_out, enc_out,trg_mask, src_mask):
        x_residual1 = dec_out

        # 1, compute multi-head attention
        x = self.mha1(q=dec_out, k=dec_out, v=dec_out, mask=trg_mask)

        # 2, add residual connection and apply layer norm
        x = self.ln1( x_residual1 + self.dropout1(x) )

        if enc_out is not None:
            # 3, compute encoder - decoder attention
            x_residual2 = x
            x = self.mha2(q=x, k=enc_out, v=enc_out, mask=src_mask)

            # 4, add residual connection and apply layer norm
            x = self.ln2( x_residual2 + self.dropout2(x) )

        # 5. positionwise feed forward network
        x_residual3 = x
        x = self.ffn(x)
        # 6, add residual connection and apply layer norm
        x = self.ln3( x_residual3 + self.dropout3(x) )

        return x

class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.decoder_emb = TransformerEmbedding(config)

        self.layers = nn.ModuleList([DecoderLayer(config)
                                     for _ in range(config.n_layer)])

        self.linear = nn.Linear(config.hidden_size, config.dec_vocab_size)

    def forward(self, dec_out, enc_out,trg_mask, src_mask):
        dec_out = self.decoder_emb(dec_out)

        for layer in self.layers:
            trg = layer(dec_out, enc_out,trg_mask,src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

    
        


class T5(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.src_pad_idx = config.src_pad_idx
        self.trg_pad_idx = config.trg_pad_idx
        self.trg_sos_idx = config.trg_sos_idx
        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        #print(src_mask.shape)
        src_mask = src_mask.cuda()
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        #print(src_trg_mask.shape)
        trg_mask1 = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) 
        trg_mask2 = self.make_no_peak_mask(trg, trg).cuda()
        #print(trg_mask1.device)
        #print(trg_mask2.device)
        trg_mask = trg_mask1*trg_mask2
        #print(trg_mask.shape)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)

        return mask
    
'''    
T5_=T5(src_pad_idx=1,
       trg_pad_idx=2,
       trg_sos_idx=3,
       enc_voc_size=50247,
       dec_voc_size=50247,
       d_model=512,
       n_head=16,
       max_len=1024,
       n_layers=6,
       drop_prob=0.1).cuda()

dummy_input = torch.randint(1,50000,(10,1024)).cuda()
label = dummy_input
output = T5_(dummy_input,dummy_input)
print(output)
output_ids = torch.argmax(output,dim=-1)
output_ids = output_ids.float()#
label = label.float()#
loss_func = nn.CrossEntropyLoss(reduction='sum')
loss = loss_func(label,output_ids)
print(loss)
'''
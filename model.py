import torch as tc
import torch.nn as nn
 
class SelfAttention(nn.Module):
    #ex. embed_size 256  heads 8  -->  head_dim = 32 
    def __init__(self,
        embed_size,  # 256   word embedding's dimension
        heads        # 8
    ):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size;  
        self.heads = heads  #heads (channel)   
        self.head_dim = embed_size // heads  #把詞向量分成8個channel，每個channel 的dimension 為32

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values_nn = nn.Linear(self.embed_size, self.embed_size,bias=False)
        self.keys_nn = nn.Linear(self.embed_size, self.embed_size,bias=False)
        self.queries_nn = nn.Linear(self.embed_size, self.embed_size,bias=False)

        self.fc_out = nn.Linear(embed_size,embed_size,bias=False)
    
    def  forward(self, queries, keys, values,   mask):
        N = queries.shape[0]  # batch size       #values  (N,seq_len,embed_size)
        value_len, key_len, query_len = values.shape[1], keys.shape[1],queries.shape[1]
        # value_len, key_len, query_len  ==  seq_len 
        
        keys = self.keys_nn(keys)            #(N,seq_len,emb_size)  32,9,256
        queries = self.queries_nn(queries)   #(N,seq_len,emb_size)  32,9,256

        values = self.values_nn(values)      #(N,seq_len,emb_size)  32,9,256   
         #*  這個NN (keys_nn, queries_nn,values_nn) 其實只是再加一層  word embedding 而已， 每個N 和 seq_len 彼此平行化  不互相影響
        
        #split embedding into self.heads pieces  (N,val_len, embed_size) --> (N,val_len, heads, head_dim)
        keys    =    keys.reshape(N, key_len, self.heads, self.head_dim)   #(32,9,8,32)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim) #(32,9,8,32)

        values  =  values.reshape(N, value_len, self.heads, self.head_dim) #(32,9,8,32)

        #QK    #attentoin_raw == energy == q*k                          
        attention_raw = tc.einsum("nqhd,nkhd->nhqk",[queries,keys]) 
        # (N,heads,q,k)   
        '''
        queries (N, query_len, heads, heads_dim)
        keys    (N, key_len, heads, heads_dim)  
        att_raw (N, heads, query_len, key_len)  # 每個query  應該對每個 key有多少的attension
                                                  每個heads (channel) 平行處理
                                                  如果 query 和 key 的 heads_dim向量相近，energy就會高
                                                  如果 query 和 key 的 heads_dim向量垂直，energy就會低
        **   Attention = softmax([Q(K.T)]/root(dk))V    
        你先想成   N和heads(channel) 都是 平行化的  
        所以實際上是  (query_len,heads_dim)*(key_len,heads_dim).T = (query_len,key_len)
        '''
        '''#* encoder case
        attentoin_raw = src_qeury * src_keys
        (N,heads,q,k)   (N,q,h,hd)  (N,k,h,hd)
        (2,8,9,9)     = (2,9,8,32)  (2,9,8,32)
        '''
        ''' #*  decoder case    (trg_to_src_attention)   
        attentoin_raw = trg_qeury * src_keys
        (N,heads,q,k)   (N,q,h,hd)  (N,k,h,hd)
        (2,8,4,9)       (2,4,8,32)  (2,9,8,32)
        '''
        if mask is not None:                       #  mask==False(padding的地方),  no attention
            attention_raw = attention_raw.masked_fill(mask==0, float("-1e20"))  # 在 mask == 0 的地方  energy(qk / weights) 填入-100000...00  這樣softmax 過後就會變成0
        '''   #*  decoder mask
        [[1 0 0 0 
          1 1 0 0
          1 1 1 0 
          1 1 1 1]]
        #*後面的還沒出來   只能對前面的人進行self  attention
        '''
        attention = tc.softmax(attention_raw/(self.embed_size**(1/2)),dim=3)  # key_len 的dimension 被softmax了
        


        out = tc.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
            N, query_len, self.embed_size)                
                          #heads*head_dim
        #attention : (N,heads, query_len, key_len)
        #values    : (N,value_len,heads, head_dim)
        #out       : (N,query_len,heads, head_dim)  #* key_len和 value_len削掉(key_len == value_len)
        
        #after flatten    (N,query_len,heads*head_dim)
        out = self.fc_out(out)   # (N,q_len, emb_size) = (emb_size, emb_size)(N,q_len, emb_size) 
        return out

class TransformerBlock(nn.Module):
    def __init__(self, 
        embed_size,         # 256   詞向量的 dimension
        heads,              # 8   we have 8 channels
        forward_expansion,   # 4
        dropout,            # 0
    ):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads) 
        self.norm1 = nn.LayerNorm(embed_size)          
        self.norm2 = nn.LayerNorm(embed_size)   #對 詞向量  進行一些 normalization

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)  #  放大之後再縮小回來  有點怪  #**  其實這也有點像是 mobile net  我覺得其中有一點 memory 的 意味
        ) 
        self.dropout = nn.Dropout(dropout)   #*  dropout probability
    def forward(self,query,key,value,mask): #(out,out,out,src_mask)  #out(2,9,256)
        '''
        out(2,9,256)    src_sentence 的詞向量
        src_mask == [
                      [[[ True,  True,  True,  True,  True,  True, False, False, False]]],
                      [[[ True,  True,  True,  True,  True,  True,  True,  True, False]]]
                    ]    shape == 2,1,1,9
        '''
        '''decoder case   trg_query, src_key, src_value
            (N,q_len, emb_size)
        ex. trg_query = q1  [[2*v1                ]     
                        q2   [0.1*v1+1.9*v2       ]                  
                        q3   [0.2*v1+0.5*v2+1.3*v3]]                 
        '''
        attention = self.attention(query,key,value,mask)  
        '''decoder case ex.
                k1  k2  k3  k4
            q1  0.1 0.6 0.2 0.1      代表  q1  (2*v1) 放在 k2 的 attention 是 0.6  
            q2  0.7 0.1 0.1 0.1   
            q3  0.1 0.2 0.5 0.2
        '''
        #(N,query_len,heads*head_dim)
        x = self.dropout(self.norm1(attention + query))   # attention ex.  0.5"吃" + 0.4 "拉麵"  + 0.1 "一起" , 要再加上自身  0.5"吃" + 0.4 "拉麵"  + (0.1+1)"一起"
                                                          # 其實不加也可以 attention 就會自己訓練成  0.3"吃" + 0.2 "拉麵"  + 0.5 "一起"  的多詞組合成單詞embedding
                                                          # 但是加了會比較好訓練(resnet) 讓他可以100%的保留自身 
        forward = self.feed_forward(x)   #feed_forward 只對  embed size nn，query_len 全部平行處理，彼此不相關
        out = self.dropout(self.norm2(forward + x))   
        return out   #(N,qeury_len,embed_size)  #ex (2,9,256)

class Encoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size,          # 德文字典詞彙量  10
        embed_size,              # 256  #**  一個單字的詞向量 dimension
        num_layers,              # 6   transformerBlock 的層數
        heads,                   # 8   we have 8 channels           
        forward_expansion,       # 4
        drop_pr,                 # 0    dropout probability
        max_length,               # 100
        device,
    ):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.device = device

        #                                 把字典裡所有的單字都變成詞向量
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.Transformer_list = nn.ModuleList(   
            [
                TransformerBlock(embed_size, heads, forward_expansion, drop_pr)
                for _ in range(num_layers)   #* num_layers == 6 -->  一共 6 層
            ]
        )
        self.dropout = nn.Dropout(drop_pr)
    def forward(self, src_x, src_mask):   
        '''
        src_x ==  [[1, 5, 6, 4, 3, 9, 0, 0, 0],
                     [1, 8, 7, 3, 4, 5, 6, 7, 0]]   shape == 2,9
        src_mask == [
                      [[[ True,  True,  True,  True,  True,  True, False, False, False]]],
                      [[[ True,  True,  True,  True,  True,  True,  True,  True, False]]]
                    ]    shape == 2,1,1,9
        '''
        N, seq_length = src_x.shape    #2,9
        position = tc.arange(0,seq_length).expand(N, seq_length).to(self.device)
        '''
            012345678
            012345678     N 個 row
            print(position.shape,"pos")  #position (2,9)
        '''
        src_emb_x = self.dropout(self.word_embedding(src_x)+self.position_embedding(position))  #將src_x 轉換成詞向量
        # print(out.shape)  #*  out  (2,9,256)   每一詞都變成了  dim 256 的向量
        for transformer_nn in self.Transformer_list:  #*  6 個 TransformerBlocks
            #          value key query
            src_emb_x = transformer_nn(src_emb_x,src_emb_x,src_emb_x,src_mask)  #transformerBlock.forward
            #out = TransformerBlock(out,out,out,src_mask)
        return src_emb_x   #(N,seq_len,embed_size)  #ex (2,9,256)

class DecoderBlock(nn.Module):
    def __init__(self,embed_size, heads, forward_expansion, drop_pr):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, forward_expansion, drop_pr)
        self.dropout = nn.Dropout(drop_pr)
    def forward(self, trg_emb_x, key, value,  src_mask, trg_mask):
        '''
          x     = trg_emb (N,trg_seq_len,embed_size) (2,4,256)
          value = enc_out (N,src_seq_len,embed_size) ex. (2,9,256) 
          key   = enc_out (N,src_seq_len,embed_size) ex. (2,9,256) 和value 相同
        '''
        '''
          src_mask == [    (2,1,1,9)
              [[[ True,  True,  True,  True,  True,  True, False, False, False]]],
              [[[ True,  True,  True,  True,  True,  True,  True,  True, False]]]
          ]
          trg_mask == [    (2,1,4,4)
              [[[1., 0., 0., 0.],
              [1., 1., 0., 0.],
              [1., 1., 1., 0.],
              [1., 1., 1., 1.]]],   same
  
          [[[1., 0., 0., 0.],
              [1., 1., 0., 0.],
              [1., 1., 1., 0.],
              [1., 1., 1., 1.]]]     same
          ]   
        '''
    #                                (option)  (essential)  pad the sentence to make it equal length           
        trg_self_attention = self.attention(trg_emb_x,trg_emb_x,trg_emb_x, trg_mask)  #*trg_x 自己先做一次 self attention
        '''   (N,q_len, emb_size)
        ex. trg_self_attention = q1  [[1*v1                ]     trg_emb_x = [[v1]
                                 q2   [0.1*v1+0.9*v2       ]                  [v2]
                                 q3   [0.2*v1+0.5*v2+0.3*v3]]                 [v3]]
        '''
        trg_query = self.dropout(self.norm(trg_self_attention+trg_emb_x))  #(N,trg_seq_len,embed_size)
        out = self.transformer_block(trg_query,key,value,src_mask)
        return out   #(N,qeury_len,embed_size) 

class Decoder(nn.Module):
    def __init__(self,
        trg_vocab_size,           # 10 英文字典詞彙量  
        embed_size,               # 256   一個單字的詞向量 dimension
        num_layers,               # 6   transformerBlock 的層數
        heads,                    # 8   we have 8 channels  
        forward_expansion,        # 4
        drop_pr,                  # 0
        max_length,               # 100
        device,
    ):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.decoder_list = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,forward_expansion,drop_pr)
                for _ in range(num_layers)   #* num_layers == 6 -->  一共 6 層
            ]   
        )

        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(drop_pr)

    def forward(self, trg_x, enc_out, src_mask, trg_mask):
        '''
        x can be trg  (N,trg_seq_len) ex (2,4) 
        enc_out (N,src_seq_len,embed_size) ex. (2,9,256)         outputs from the encoder
        '''
        N, seq_length = trg_x.shape
        positions = tc.arange(0, seq_length).expand(N, seq_length).to(self.device)
        '''#position (2,4)
            0123
            0123    N 個 row
            print(position.shape,"pos")  
        '''
        trg_emb_x = self.dropout((self.word_embedding(trg_x)+self.position_embedding(positions)))
        #trg_emb  (2,4,256)
        for decoder_nn in self.decoder_list:
            #                       query      key     value    
            trg_emb_x = decoder_nn(trg_emb_x,enc_out, enc_out,src_mask,trg_mask)  
        out = self.fc_out(trg_emb_x) 
        return out

class Transformer(nn.Module):
    def __init__(self,
        embed_size,
        src_vocab_size,   # 字典詞彙量
        trg_vocab_size,   # 字典詞彙量
        src_pad_idx,      #pad 對應的 空index -->  0
        trg_pad_idx,      #pad 對應的 空index -->  0
        heads, 
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        drop_pr,         #*  dropout probability 
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size,
                        num_encoder_layers, heads, forward_expansion,
                        drop_pr, max_len,device)

        self.decoder = Decoder(trg_vocab_size, embed_size,
                        num_decoder_layers, heads, forward_expansion, 
                        drop_pr, max_len, device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        #            N,src_max_len  (2,9)                 equals to reshape(N,1,1,src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # reshape 2,9  to (2,1,1,9)
        # (N,1,1,src_max_len)   # src_pad idx set to 0, otherwise set to 1
        # print(src_mask)
        return src_mask.to(self.device)  
        '''
        src_mask == [
                      [[[ True,  True,  True,  True,  True,  True, False, False, False]]],
                      [[[ True,  True,  True,  True,  True,  True,  True,  True, False]]]
                    ]
        '''
    def make_trg_mask(self, trg):
        # print(trg.shape,"trg.shape")
        #* make a triangular matrix
        N,trg_max_len = trg.shape  # 2, 4
        #                          4*4 triangle  repeat 2 times
        trg_mask = tc.tril(tc.ones((trg_max_len,trg_max_len))).expand(  #重複 N 次  每個 N 都是同一個mask
            N,1,trg_max_len,trg_max_len
        )
        # print(trg_mask)
        return trg_mask.to(self.device)
        '''
        trg_mask == [
                     [[[1., 0., 0., 0.],
                       [1., 1., 0., 0.],
                       [1., 1., 1., 0.],
                       [1., 1., 1., 1.]]],   same

                    [[[1., 0., 0., 0.],
                      [1., 1., 0., 0.],
                      [1., 1., 1., 0.],
                      [1., 1., 1., 1.]]]     same
                   ]   
        '''        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        #enc_src  #(N,seq_len,embed_size)  #ex (2,9,256)  #N, seq
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

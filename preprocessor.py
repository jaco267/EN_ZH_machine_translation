import os
import download_data
from torchtext.data.utils import get_tokenizer
import pandas as pd
from collections import Counter
import random
from torchtext.vocab import vocab
import torch as tc
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class Preprocessor():
    def __init__(self,batch_size):
       self.data_folder = "./data"
       self.train_test_folder = f'{self.data_folder}/train_test_data'; 
       
       self.__load_train_test_csv_file()   ### create train_data.csv (if not exist)
       print("--create tokenizer--")
       self.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
       self.zh_tokenizer = get_tokenizer('spacy', language='zh_core_web_sm')
       self.seed = 3
       print("--read data from csv--")
       train_df = pd.read_csv(f'{self.train_test_folder}/train_data.csv')
       val_df   = pd.read_csv(f"{self.train_test_folder}/validation_data.csv") 
       train_en_sentence_list = [sent.lower() for sent in train_df['en'].values.tolist()] 
       train_zh_sentence_list = train_df['zh'].values.tolist()
       val_en_sentence_list = [sent.lower() for sent in val_df['en'].values.tolist()] 
       val_zh_sentence_list   =   val_df['zh'].values.tolist()
       print("---building EN/ZH dictionary---")
       self.en_vocab = self.__build_vocab(train_en_sentence_list, self.en_tokenizer)
       self.en_vocab.set_default_index(self.en_vocab['<unk>'])

       self.zh_vocab = self.__build_vocab(train_zh_sentence_list, self.zh_tokenizer)
       self.zh_vocab.set_default_index(self.zh_vocab['<unk>'])
       print("---building train_data / test_data---")
       self.train_data= self.__sent2tensor(train_en_sentence_list, train_zh_sentence_list)
       def sort_by_src_len(e):
           """train data will be sort by src len (to compress the tensor size)"""
           src,trg = e     
           src_len = src.shape[0]  
           return src_len
       self.train_data.sort(key = sort_by_src_len) #sorting by english sentence length 
                                                    # so we can save space after batchify the tensor
       self.val_data   = self.__sent2token(val_en_sentence_list, val_zh_sentence_list)
       print("---building train_data iterator---")

       self.train_iter = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, collate_fn=self.__generate_batch)
    def get_vocab(self):
        return self.en_vocab,self.zh_vocab
    def get_tokenizer(self):
        return self.en_tokenizer,self.zh_tokenizer
    def get_train_iter(self):
        return self.train_iter
    def get_train_test_data(self):
        return self.train_data,self.val_data
    def __load_train_test_csv_file(self):
        if not os.path.exists(self.train_test_folder):    
           os.makedirs(self.train_test_folder)
        if os.path.exists("./data/train_test_data/train_data.csv"):
           print("train datafile exist")
        else:
           print("train datafile doesn't exist")
           print("start downloading data")
           download_data.make_train_test_data_csv_file(self.data_folder,self.train_test_folder)
    def __build_vocab(self,sentence_list, tokenizer):
        counter = Counter()
        for string_ in sentence_list:
            counter.update(tokenizer(string_))
        #shuffle the counter  
        vocab_list = []   
        for k in counter.keys():
            if(counter[k]>1):    vocab_list.append(k)   #*  min_freq =2
        random.Random(self.seed).shuffle(vocab_list)  # random.shuffle(vocab_list)
        #set seed 3
        counter2 = Counter()
        counter2.update(vocab_list)
        return vocab(counter2, 
                    specials=['<unk>', '<pad>', '<sos>', '<eos>'],
                    min_freq=1)
    def __sent2tensor(self,en_sent_list,zh_sent_list,
            min_len=5,   # discard sentences that is too short 
            max_len=50,  # discard sentences that is too long
            minus_min=-6,  #If there is too much difference in length between Chinese and English,
            minus_max=8):  # it will be removed   (-6<en-zh<8)
        """convert sentences to tensor\n
          when training, we need to use torch.Tensor type data """
        data = []
        for (raw_en, raw_zh) in zip(en_sent_list, zh_sent_list):
            en_tensor_ = tc.tensor([self.en_vocab[token] for token in self.en_tokenizer(raw_en)],
                                    dtype=tc.long)
            zh_tensor_ = tc.tensor([self.zh_vocab[token] for token in self.zh_tokenizer(raw_zh)],
                                    dtype=tc.long)
            en_seq_len,zh_seq_len =  int(en_tensor_.shape[0]),int(zh_tensor_.shape[0])

            if min_len<en_seq_len<max_len and min_len<zh_seq_len<max_len:   #*把太長或太短的句子丟掉，圖表顯示  大部分的句子長度都落在10~40
              if minus_min<=(en_seq_len-zh_seq_len)<=minus_max:                           #* 有時候中英文長度差太多   我們會想要讓兩者相近  這樣子sort的時候 matrix 的padding 才會比較小
                  data.append((en_tensor_,zh_tensor_))   
        return data

    def __sent2token(self,en_sent_list,zh_sent_list,
            min_len=5,max_len=50,minus_min=-6,minus_max=8):
        """convert sentences to integer list (token list)\n
          when testing, we need to use int[] type data """
        data = []
        for (raw_en,raw_zh ) in zip( en_sent_list,zh_sent_list):
            en_tokens = [token for token in self.en_tokenizer(raw_en)]                       
            zh_tokens = [token for token in self.zh_tokenizer(raw_zh)]
            en_seq_len,zh_seq_len =  len(en_tokens),len(zh_tokens)
            if min_len<en_seq_len<max_len and min_len<zh_seq_len<max_len:   
                if minus_min<=(en_seq_len-zh_seq_len)<=minus_max: 
                    data.append((en_tokens,zh_tokens))   
        return data
    def __generate_batch(self,data_batch):
        PAD_IDX = self.en_vocab['<pad>']; 
        BOS_IDX = self.en_vocab['<bos>'];
        EOS_IDX = self.en_vocab['<eos>'];
        en_batch,zh_batch = [], []
        for (en_item,zh_item) in data_batch:
            en_batch.append(tc.cat([tc.tensor([BOS_IDX]), en_item, tc.tensor([EOS_IDX])], dim=0))
            zh_batch.append(tc.cat([tc.tensor([BOS_IDX]), zh_item, tc.tensor([EOS_IDX])], dim=0))
        # print(len(en_batch),"???",en_batch[0].shape,en_batch[-1].shape)  #len(en_batch):128,  en_batch[0].shape: [15],en_batch[-1].shape: [16]
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX,batch_first=True)  #(batch_size,seq_len)
        zh_batch = pad_sequence(zh_batch, padding_value=PAD_IDX,batch_first=True)
        # print(en_batch.shape,"...")  # 128,16
        return en_batch,zh_batch, 
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
def plot_src_trg_sentence_len(en_len_count,zh_len_count,en_seq_len_list,zh_seq_len_list):
    plt.xlabel('sentence len');  plt.ylabel('times')
    plt.xlim([0, 60])
    indexes, values = zip(*en_len_count.items());  plt.bar(indexes, values, width=1)
    indexes, values = zip(*zh_len_count.items());  plt.bar(indexes, values, width=1)
    plt.show()

    print("---分析 英文 中文 句子長度 差  (en-zh)---")
    #* 資料有時候中英文長度差太多(壞資料)   我們會想要讓兩者相近  
    minus_len = np.array(en_seq_len_list) - np.array(zh_seq_len_list)  
    en_len_count = Counter(minus_len)
    indexes, values = zip(*en_len_count.items())
    plt.xlim([-10, 15])
    plt.xlabel('sentence len |en-zh|');    plt.ylabel('times');
    plt.bar(indexes, values, width=1)
    plt.show()


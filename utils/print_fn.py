def print_tensor_to_str(sentence_tensor,vocab,end="\n"):
    sen_list = sentence_tensor.tolist()
    str_list = vocab.lookup_tokens(sen_list)
    sentence_str = '|'.join(str_list)
    print(sentence_str,end=end)
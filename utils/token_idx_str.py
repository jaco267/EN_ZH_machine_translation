#%%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wordList_to_tensor(wordList,vocab):
    wordList.insert(0, '<bos>')  # Add <SOS> and <EOS> in beginning and end respectively
    wordList.append('<eos>')    
    # ['<bos>', 'ein', 'boot',... ,'ufer', 'gezogen', '.', '<eos>']
    word_idx = [vocab[token] for token in wordList]  # Go through each word and convert to an index
    #[2, 5, 194,... , 547, 933, 4, 3]
    sentence_tensor = torch.LongTensor(word_idx).reshape(1,len(wordList)).to(device)    # Convert to Tensor
    return sentence_tensor  #shape  [1,17] # [N, seq_length]
def test_translate(sentence,model,src_tokenizer,src_vocab,trg_vocab):
    #testing the model (translate one german sentence)
    wordList = [token.lower() for token in src_tokenizer(sentence)] #everything in lower case (which is what our vocab is)
    sentence_tensor = wordList_to_tensor(wordList,src_vocab)
    #shape  [1,17]  # [N,seq_length]
    translated_sentence = translate_sentence(
        model,sentence_tensor, trg_vocab, max_length=50
    )
    print(f"Translated example sentence: \n {translated_sentence}")
    return translated_sentence

def translate_sentence(model, sentence_tensor,trg_vocab, max_length=50):
    # ex. sentence_tensor (1,17)
    outputs = [trg_vocab["<bos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).reshape(1,len(outputs)).to(device)
        #i=1 (1,1), i=2 (1,2), ....  ,i=seq_len  (1,seq_len)
        #ex. i=9 trg_tensor = (1,9)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)
        #output (9,1,output_size)
        #  1,9,out_size
        best_guess = output.argmax(dim=2)[0, -1].item()  #(9,1,out)->(9,1)->(1,1)->1
        outputs.append(best_guess)

        if best_guess == trg_vocab["<eos>"]:
            break

    translated_sentence = [trg_vocab.lookup_token(idx) for idx in outputs]
    # remove start token
    return translated_sentence[1:]



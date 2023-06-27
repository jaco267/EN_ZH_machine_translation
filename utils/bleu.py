
from utils.token_idx import test_translate,translate_sentence, wordList_to_tensor
from torchtext.data.metrics import bleu_score
def bleu(data, model, src_vocab, trg_vocab):
    print("caculating bleu score")
    targets = []
    outputs = []
    for en_str,zh_str in data:
        en_str2 = [token.lower() for token in en_str]  #好像可以comment掉
        en_tensor = wordList_to_tensor(en_str2,src_vocab)
        # prediction = translate_sentence(model, src, german, english, device)
        pred_str = translate_sentence(
            model,en_tensor, trg_vocab, max_length=50
        )
        #predict string
        pred_str = pred_str[:-1]  # remove <eos> token

        outputs.append(pred_str)
        targets.append([zh_str])  #bleu score target 可能有多個  所以會多一層[]
    return bleu_score(outputs, targets)
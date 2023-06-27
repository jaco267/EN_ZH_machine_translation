import torch as tc
import torch.nn as nn
import torch.optim as optim
from utils.token_idx import test_translate
from utils.bleu import bleu
from model import Transformer
from preprocessor import Preprocessor
import os
import fire

def main(batch_size = 32,num_epochs = 12,      
         learning_rate = 3e-4,load_model = False,
         ckpt_file = "checkpoint_epoch3.pth.tar"
    ):
    checkpoint_folder = './checkpoint'
    checkpointFile = f"{checkpoint_folder}/{ckpt_file}";
    if not os.path.exists(checkpoint_folder):    os.makedirs(checkpoint_folder)
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')


    Pre = Preprocessor(batch_size)
    en_vocab,zh_vocab = Pre.get_vocab();
    en_tokenizer,_ = Pre.get_tokenizer();
    train_iter = Pre.get_train_iter()
    _, val_data = Pre.get_train_test_data()

    print("train_iter lengh(total number of batch)",len(train_iter)) #314
    print(f"data size(total number of sentences) ~= {batch_size*len(train_iter)}")   #40192

    loss_list,translate_list,score_list,former_epochs = [], [], [], 0
    model = Transformer(
        embed_size=512,
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(zh_vocab),
        src_pad_idx= en_vocab['<pad>'],
        trg_pad_idx= zh_vocab['<pad>'],
        heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        forward_expansion=4,
        drop_pr=0.1,
        max_len=100,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                        factor=0.1, patience=10, verbose=True)  #10個 iteration效能沒改善的話 就把Learning rate*0.1

    criterion = nn.CrossEntropyLoss(ignore_index = en_vocab["<pad>"]) #ignore pad index
    if load_model:
        print(f"=> Loading {checkpointFile}")
        state_dic_load, optimizer_load,scheduler_load,former_epochs\
                ,loss_list,translate_list,score_list = tc.load(checkpointFile).values()
        model.load_state_dict(state_dic_load)
        optimizer.load_state_dict(optimizer_load)
        scheduler.load_state_dict(scheduler_load)
        print('score list length',len(score_list))

    #  	坏消息是，在确保金融大企业实际可破产方面几乎没有什么进展。
    sentence = "The bad news is that there has been almost no progress\
    in terms of ensuring that large financial firms actually can go bankrupt."
    #"ein pferd geht unter einer brücke neben einem boot."
    model.eval()  #   model is in testing mode
    translate_sentence0 = test_translate(sentence,model,en_tokenizer,en_vocab,zh_vocab)
    score = bleu(val_data[0:10], model, en_vocab, zh_vocab)
    print(f"Bleu score {score*100:.2f}")

    for epoch in range(former_epochs+1,former_epochs+1+num_epochs):
        print(f"[Epoch {epoch} / {former_epochs + num_epochs}]")
        model.train()
        losses = []
        for batch_idx, (en,zh) in enumerate(train_iter):
            # Get input and targets and get to cuda
            inp_data = en.to(device)   #ex.(32,17)
            target = zh.to(device)     #ex.(32,24)
            # Forward prop
            output = model(inp_data, target[:, :-1])  #trg (24,32) --> (23,32) without eos
            output = output.reshape(-1, output.shape[2]) #(trg_len, N, output_size) -> (trg_len*N, output_size)
            target = target[:,1:].reshape(-1)  #(trg_len,N) ->(trg_len*N)
            optimizer.zero_grad()
            loss = criterion(output, target)
            losses.append(loss.item())

            loss.backward()  # Back prop
            tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Clip to avoid exploding gradient issues
            optimizer.step()  # Gradient descent step

            loss_list.append(loss.cpu().item())
            
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

        if epoch % 1 == 0 :
            model.eval()  #   model is in testing mode
            # test translate
            translate_sentence0 = test_translate(sentence,model,en_tokenizer,en_vocab,zh_vocab)
            translate_list.append(translate_sentence0)
            
            # bleu score
            score = bleu(val_data[1:100], model, en_vocab, zh_vocab)
            print(f"Bleu score {score*100:.2f}")
            score_list.append(round(score,2))
            if epoch % 3 == 0 :
                print("=> Saving checkpoint")
                checkpoint = {"state_dict": model.state_dict(), 
                            "optimizer": optimizer.state_dict(),
                            "schedular": scheduler.state_dict(),
                            "former_epochs": epoch,"loss_list":loss_list,
                            "translate_list":translate_list,"score_list":score_list}
                tc.save(checkpoint, f"{checkpoint_folder}/checkpoint_epoch{epoch}.pth.tar")

    import matplotlib.pyplot as plt
    import numpy as np

    def plot_2d_graph(y_raw,title,x_name,y_name,ax):
        X=np.arange(len(y_raw))
        Y=np.array(y_raw)
        ax.plot(X,Y,label = 'loss')
        ax.set_title(title,fontsize=20,\
            weight='bold',style='italic')
        ax.set_xlabel(x_name,fontsize=20)
        ax.set_ylabel(y_name,fontsize=20)
        ax.tick_params(axis='y', labelrotation=0)

    fig,ax=plt.subplots(2,1,figsize=(8,8))
    ax[0].set_ylim([0,10])
    plot_2d_graph(loss_list,"training curve","iterations","loss",ax[0])
    ax[1].set_ylim([0,max(score_list)+0.03])
    plot_2d_graph(score_list,"bleu score","epochs","score",ax[1])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)










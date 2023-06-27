#%%
import pandas as pd
import wget
import os
import gzip, shutil 

def _download_data_file(file_path):
    if not os.path.exists(file_path):
        print(f"can't find {file_path}")
        print(f"installing {file_path}")
        news_zip_file_path = f"{file_path}.gz"
        if not os.path.exists(news_zip_file_path):
            print("downloading zip file")
            wget.download("https://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-zh.tsv.gz",
                        news_zip_file_path)
        with gzip.open(news_zip_file_path,"rb") as infile:
            with open(file_path,"wb") as outfile:
                shutil.copyfileobj(infile,outfile)
    else:
        print(f"{file_path} exists. ")
def _add_field_name(news_data_file_path):
    with open(news_data_file_path, "r+") as f:
        content = f.readline().strip("\n")
        if (content != "en\tzh"):
            print(content)
            print("didn't canonicalize")
            print("tuning tsv file in to canonical form...")
            content = f.read()   ## read all files content
            f.seek(0, 0)         # move the cursor to the beginning
            f.write("en\tzh" + '\n' + content)  # write "en  zh" as tsv file's field name
        else:
            print(content)
            print("tsv file is in canonical form.")
def _tsv_to_csv(tsv_data_file_path,train_test_folder):
    raw_df = pd.read_csv(tsv_data_file_path,sep='\t', on_bad_lines='skip') #他好像會一次skip十行  可以改成'warn'來檢查看看   原始長度 #306330  raw_df長 306140
    clean_df=raw_df.dropna()   #移除nan   clean_df長297919

    shuffle_df = clean_df.sample(frac=1,random_state=1).reset_index(drop=True)

    print("---把資料切割成  train data, validation data---")
    all_len = len(shuffle_df)      #297919
    train_len = int(all_len*0.20)  #59583
    val_len   = int(all_len*0.01)  #2979
    train_df = shuffle_df.iloc[:train_len,:]
    val_df   = shuffle_df.iloc[train_len:train_len+val_len,:].reset_index(drop=True)
    print(f"---saving data to {train_test_folder}/train_data.csv  && {train_test_folder}/validation_data.csv---")
    train_df.to_csv(f"{train_test_folder}/train_data.csv", index=False,encoding='utf-8')
    val_df.to_csv(f"{train_test_folder}/validation_data.csv", index=False,encoding='utf-8')  #header=False,index=False




def make_train_test_data_csv_file(data_folder,train_test_folder):
    news_data_file_path = f"{data_folder}/news-commentary-v14.en-zh.tsv"
    _download_data_file(news_data_file_path)
    _add_field_name(news_data_file_path)
    _tsv_to_csv(news_data_file_path,train_test_folder)
    



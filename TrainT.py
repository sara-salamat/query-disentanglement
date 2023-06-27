import logging
import os
import pathlib
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from Model import TransformerPredictor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    )

torch.cuda.empty_cache()
path_to_data = pathlib.Path('data')


# model_name = 'microsoft/MiniLM-L12-H384-uncased'
# model_name = 'microsoft/deberta-v3-base'
model_name = '' # Put the encoder model from Huggingface model hub

data = pd.read_csv(path_to_data/'classification_pairs_420k_420k.csv', header=None)
data = data.dropna()


model_save_path = f'output/qpp-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
model_save_path = pathlib.Path(model_save_path)
os.makedirs(model_save_path, exist_ok=True)
shutil.copyfile(__file__, model_save_path/'TrainScript.py')
shutil.copyfile('Model.py', model_save_path/'ModelScript.py')
emb_size=768 ## BERT, deBERTa



train,val = train_test_split(data, test_size=0.2)


print('train val splitted')

class MyDataset(Dataset):
    def __init__(self, initial_q, target_q, initial_score: list, target_score: list, labels: list) -> None:
        super().__init__()
        self.initial_q = initial_q
        self.target_q = target_q
        self.initial_score = initial_score
        self.target_score = target_score
        self.sim_labels = labels
        

    
    def __len__(self):
        return len(self.initial_score)

    def __getitem__(self, idx):

        initial_q_item = {'input_ids': self.initial_q['input_ids'][idx],
                          'attention_mask': self.initial_q['attention_mask'][idx],
                          }

        target_q_item = {'input_ids': self.target_q['input_ids'][idx],
                         'attention_mask': self.target_q['attention_mask'][idx],
                          }

        similarity_label = self.sim_labels[idx]
        return initial_q_item, tensor(self.initial_score[idx]),target_q_item, tensor(self.target_score[idx]), similarity_label


tokenizer = AutoTokenizer.from_pretrained(model_name)
initial_q_train_encodings = tokenizer(train[1].to_list(),return_tensors='pt', padding=True)
target_q_train_encodings = tokenizer(train[3].to_list(),return_tensors='pt', padding=True)
initial_q_val_encodings = tokenizer(val[1].to_list(),return_tensors='pt', padding=True)
target_q_val_encodings = tokenizer(val[3].to_list(),return_tensors='pt', padding=True)

TD = MyDataset(initial_q_train_encodings, target_q_train_encodings, train[2].to_list(), train[4].to_list(),train[5].to_list())
VD = MyDataset(initial_q_val_encodings, target_q_val_encodings, val[2].to_list(), val[4].to_list(),val[5].to_list())



batch_size = 32
TrainData = DataLoader(TD, batch_size=batch_size, shuffle=True, num_workers=5)
ValidationData = DataLoader(VD, batch_size=batch_size, shuffle=True, num_workers=5)


latent_space_size = emb_size
content_portion_size = 500
model_ = TransformerPredictor(
    input_size=emb_size ,
    latent_space_size=latent_space_size, 
    content_portion_size=content_portion_size,
    model_name=model_name
    )




if torch.cuda.is_available():
    model_ = model_.cuda()

logging.info("---Training loop---")



lr = 0.001
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_.parameters()),lr=lr)     
epochs = 5

def to_cuda(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda')
    token_type_ids = Tokenizer_output['token_type_ids'].to('cuda')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda')
    output = {'input_ids' : tokens_tensor, 
              'token_type_ids' : token_type_ids, 
              'attention_mask' : attention_mask}
                  
    return output
for e in range(epochs):
    train_loss = 0.0
    model_.train()
    for initial_q, initial_score, target_q, target_score, similarity_label in tqdm(TrainData):
        if torch.cuda.is_available():
            initial_q, initial_score, target_q, target_score, similarity_label = to_cuda(initial_q), initial_score.cuda(), to_cuda(target_q), target_score.cuda(), similarity_label.cuda()
            
        optimizer.zero_grad(set_to_none=True)
        total_loss, predicted_difference_score = model_(initial_q, initial_score, target_q, target_score, similarity_label)
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
    model_.eval()
    val_loss = 0.0
    with torch.no_grad():
        for initial_q, initial_score, target_q, target_score, similarity_label in tqdm(ValidationData):

            if torch.cuda.is_available():
                initial_q, initial_score, target_q, target_score, similarity_label = to_cuda(initial_q), initial_score.cuda(), to_cuda(target_q), target_score.cuda(), similarity_label.cuda()
            total_loss, predicted_difference_score = model_(initial_q, initial_score, target_q, target_score, similarity_label)
            val_loss += total_loss

    
    print(f'Epoch {e+1} \t Training Loss: {train_loss / len(TrainData)} \t Validation Loss: {val_loss / len(ValidationData)}')
    ep_model_name = 'model_ep'+str(e)+'.pkl'
    with open(model_save_path/ep_model_name, 'wb') as f_:
        torch.save(model_, f_)



with open(model_save_path/'model.pkl', 'wb') as f:
    torch.save(model_, f)

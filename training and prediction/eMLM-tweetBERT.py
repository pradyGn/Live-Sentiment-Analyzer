import transformers
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer
import torch
import pandas as pd
import os
import json
from tensorflow.keras.utils import to_categorical
import torch.nn as nn
import numpy as np
import random
from transformers import AdamW
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = BertForMaskedLM.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')

model.to(device)

with open('/content/drive/MyDrive/finBERT_deployment/tweets_data.txt') as f:
    lines = f.readlines()

inputs = tokenizer(lines, return_tensors='pt', max_length=64, truncation=True, padding='max_length')

inputs.keys()

inputs['labels'] = inputs.input_ids.detach().clone()

lexicon_data = pd.read_csv("/content/drive/MyDrive/finBERT_deployment/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep='\t', names=['word','emotion','label'])

def getwords(lexicon_data):
  words = {}

  w = list(lexicon_data['word'])


  l = list(lexicon_data['label'])
  e = list(lexicon_data['emotion'])

  for i in range(len(l)):
      if l[i] == 1:
          if l[i] not in words:
              words[w[i]] = [e[i]]
          else:
              words[w[i]].append(e[i])
  return words

words = getwords(lexicon_data)

k = 0.5

#creating masks

for i in range(len(inputs.input_ids)):
    emo_selection = []
    example_len = 0
    for j in range(len(inputs.input_ids[i])):
        if inputs.input_ids[i][j] == torch.tensor(102):
            example_len = j + 1
            break
        decoded_word = tokenizer.decode(inputs.input_ids[i][j])
        decoded_word = decoded_word.replace(' ', '')
        if decoded_word in words:
            if random.uniform(0, 1) < k:
                emo_selection.append(j)
    inputs.input_ids[i, emo_selection] = 103
    if (example_len - len(emo_selection)) == 0:
        print("Den is 0 at: ")
        print(i)
        break
    remain_prob = max((example_len*0.15 - len(emo_selection)*k)/(example_len - len(emo_selection)), 0)
    if remain_prob > 0:
        rand = torch.rand(example_len)
        mask_arr = (rand < remain_prob) * (inputs.input_ids[i][:example_len] != 101) * (inputs.input_ids[i][:example_len] != 102) * (inputs.input_ids[i][:example_len] != 103)
        gen_selection = torch.flatten(mask_arr.nonzero()).tolist()
        inputs.input_ids[i, gen_selection] = 103

class TweetsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

tweetsdataset = TweetsDataset(inputs)

tweetsDataLoader = torch.utils.data.DataLoader(tweetsdataset, batch_size=16, shuffle=True)

model.train()
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

def train(model, epochs, DataLoader, optim):
  for epoch in range(epochs):
      loop = tqdm(DataLoader, leave=True)
      for batch in loop:
          optim.zero_grad()
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
          loss.backward()
          optim.step()
          loop.set_description(f'Epoch {epoch}')
          loop.set_postfix(loss=loss.item())

train(model, 2, tweetsDataLoader, optim)

#!python3 "/content/drive/MyDrive/finBERT_deployment/datasets.py" --data_path "/content/drive/MyDrive/finBERT_deployment/Sentences_66Agree.txt"

#train = pd.read_csv("/content/data/sentiment_data/train.csv", encoding="iso-8859-1", sep=' 	', names=['text','label'])

#X_train = train['text']
#y_train = train['label']

#val = pd.read_csv("/content/data/sentiment_data/test.csv", encoding="iso-8859-1", sep=' 	', names=['text','label'])

#X_val = val['text']
#y_val = val['label']

train = pd.read_csv("/content/drive/MyDrive/finBERT_deployment/labelled_tweets_2.csv", encoding='latin-1')

train.columns = ['label', 'idc0','time', 'idc1', 'idc2', 'tweet']

X_train, X_test, y_train, y_test = train_test_split(train['tweet'], train['label'], test_size=0.2, random_state=42, shuffle=True)

X_train[10:20]

X_train = list(X_train)
X_test = list(X_test)

def retlist(iter):
  text = list(iter)
  newtext = []
  for line in text:
      line = line.split("\t")[1]
      newtext.append(line)

  import copy
  text = copy.deepcopy(newtext)
  return text

def getlabels(iter):
  label_lis = []
  for lab in iter:
    if lab == 'negative':
      label_lis.append(0)
    elif lab == 'neutral':
      label_lis.append(1)
    else:
      label_lis.append(2)
  return label_lis

#X_train = retlist(X_train)
#X_val = retlist(X_val)

train_inputs = tokenizer(X_train, max_length=64, truncation=True, padding='max_length')
test_inputs = tokenizer(X_test, max_length=64, truncation=True, padding='max_length')

train_inputs.keys()

train_inputs.input_ids = torch.Tensor(train_inputs.input_ids).long()
train_inputs.token_type_ids = torch.Tensor(train_inputs.token_type_ids).long()
train_inputs.attention_mask = torch.Tensor(train_inputs.attention_mask).long()

test_inputs.input_ids = torch.Tensor(test_inputs.input_ids).long()
test_inputs.token_type_ids = torch.Tensor(test_inputs.token_type_ids).long()
test_inputs.attention_mask = torch.Tensor(test_inputs.attention_mask).long()

#y_train = getlabels(y_train)
#y_test = getlabels(y_val)

y_train = list(y_train)
y_test = list(y_test)

for i in range(len(y_train)):
  if y_train[i] == 4:
    y_train[i] = 1

for i in range(len(y_test)):
  if y_test[i] == 4:
    y_test[i] = 1

Trainlabel_tensor = torch.Tensor(y_train).long()
Testlabel_tensor = torch.Tensor(y_test).long()

train_inputs['labels'] = Trainlabel_tensor
test_inputs['labels'] = Testlabel_tensor

max_len = max([len(sent) for sent in test_inputs.input_ids])
print('Max length: ', max_len)

train_data = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_inputs.labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

test_data = TensorDataset(test_inputs.input_ids, test_inputs.attention_mask, test_inputs.labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)


class BertClassifier(nn.Module):
     def __init__(self, freeze_bert=True):
         super(BertClassifier, self).__init__()
         D_in, H, D_out = 64001, 256, 4 #768, 256, 3
 
         self.bert = model
 
         self.classifier = nn.Sequential(
             nn.Dropout(0.1),
             nn.ReLU(),
             nn.Linear(D_in, H),
             nn.ReLU(),
             nn.Dropout(0.1),
             nn.Linear(H, D_out)
         )
 
         if freeze_bert:
             for param in self.bert.parameters():
                 param.requires_grad = False
         
     def forward(self, input_ids, attention_mask):
         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
         
         last_hidden_state_cls = outputs[0][:, 0, :]
 
         logits = self.classifier(last_hidden_state_cls)
 
         return logits

from transformers import AdamW, get_linear_schedule_with_warmup

def initialize_model(epochs=4):
    bert_classifier = BertClassifier(freeze_bert=True)

    bert_classifier.to(device)

    optimizer = AdamW(bert_classifier.parameters(),
                      lr=2e-5,    
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

import random
import time

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        #training

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        
        #evaluation
        if evaluation == True:
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    model.eval()

    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    predictions = []

    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        #print(logits)
        preds = torch.argmax(logits, dim=1).flatten()

        predictions.append(preds)

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return predictions, val_loss, val_accuracy

set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=6)
train(bert_classifier, train_dataloader, test_dataloader, epochs=6)

evaluate(bert_classifier, test_dataloader)

#saving model
model.dump(bert_classifier, open("bert_classifier.pkl", "wb"))




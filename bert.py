# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:53:07 2019

@author: seungjun
"""

#skt kobert model using
import pandas as pd
import numpy as np
from mxnet.gluon import nn, rnn
from mxnet import gluon, autograd
import gluonnlp as nlp
from mxnet import nd 
import mxnet as mx
import time
import itertools
import random

from kobert.mxnet_kobert import get_mxnet_kobert_model
from kobert.utils import get_tokenizer

import pandas as pd
import numpy as np

test=pd.read_csv(r'public_test.csv',encoding='utf-8')
train=pd.read_csv(r'train.csv',encoding='utf-8')
train=train[['text','smishing']]

n_of_train=int(len(train)*0.8)
n_of_val=int(len(train)-n_of_train)
print(n_of_train)
print(n_of_val)
train1=np.asarray(train[:n_of_train])
valid1=np.asarray(train[n_of_train:])
train2=(train[:n_of_train])
valid2=(train[n_of_train:])


print('start')
ctx=mx.gpu()
bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx)



#token----------------------------------------------------------------
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
ds = gluon.data.SimpleDataset([['나는 이승준입니다.', '과제가 너무 많은걸요?']])
trans = nlp.data.BERTSentenceTransform(tok, max_seq_length=10)
list(ds.transform(trans))
print('token complete')

#Dataset----------------------------------------------------------------
class BERTDataset(mx.gluon.data.Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        sent_dataset = gluon.data.SimpleDataset([[
            i[sent_idx],
        ] for i in dataset])
        self.sentences = sent_dataset.transform(transform)
        self.labels = gluon.data.SimpleDataset(
            [np.array(np.int32(i[label_idx])) for i in dataset])

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
    
max_len=128

data_train=BERTDataset(train1, 0,1,tok,max_len,True,False)
data_valid=BERTDataset(valid1, 0,1,tok,max_len,True,False)    
print('make dataset')
#Classifier----------------------------------------------------------------  
class BERTClassifier(nn.Block):
    def __init__(self,
                 bert,
                 num_classes=2,
                 dropout=None,
                 prefix=None,
                 params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def forward(self, inputs, token_types, valid_length=None):
        _, pooler = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler)
    
    
model=BERTClassifier(bert_base, num_classes=2, dropout=0.1)

model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
model.hybridize()

loss_function=gluon.loss.SigmoidBinaryCrossEntropyLoss()
metric=mx.metric.Accuracy()

batch_size=32
lr=5e-5
train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = mx.gluon.data.DataLoader(data_valid, batch_size=batch_size, num_workers=5)

trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                        {'learning_rate': lr, 'epsilon': 1e-9, 'wd':0.01})

log_interval = 4
num_epochs = 5


for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
    v.wd_mult = 0.0
params = [
    p for p in model.collect_params().values() if p.grad_req != 'null'
]

def evaluate_accuracy(model, data_iter, ctx=ctx):
    acc = mx.metric.Accuracy()
    i = 0
    for i, (t,v,s, label) in enumerate(data_iter):
        token_ids = t.as_in_context(ctx)
        valid_length = v.as_in_context(ctx)
        segment_ids = s.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(token_ids, segment_ids, valid_length.astype('float32'))
        acc.update(preds=output, labels=label)
        if i > 1000:
            break
        i += 1
    return(acc.get()[1])
    
    
accumulate = 4
step_size = batch_size * accumulate if accumulate else batch_size
num_train_examples = len(data_train)
num_train_steps = int(num_train_examples / step_size * num_epochs)
warmup_ratio = 0.1
num_warmup_steps = int(num_train_steps * warmup_ratio)
step_num = 0

all_model_params = model.collect_params()


if accumulate and accumulate > 1:
    for p in params:
        p.grad_req = 'add'

for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            non_warmup_steps = step_num - num_warmup_steps
            offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
            new_lr = lr - offset * lr
        trainer.set_learning_rate(new_lr)
        with mx.autograd.record():
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float'))
            ls = loss_function(out, label).mean()

        # backward computation
        ls.backward()
        if not accumulate or (batch_id + 1) % accumulate == 0:
          trainer.allreduce_grads()
          nlp.utils.clip_grad_global_norm(params, 1)
          trainer.update(accumulate if accumulate else 1)
          step_num += 1
          if accumulate and accumulate > 1:
              # set grad to zero for gradient accumulation
              all_model_params.zero_grad()

        step_loss += ls.asscalar()
        metric.update([label], [out])
        if (batch_id + 1) % (50) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.10f}, acc={:.3f}'
                         .format(epoch_id + 1, batch_id + 1, len(train_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0
    test_acc = evaluate_accuracy(model, test_dataloader, ctx)
    print('Test Acc : {}'.format(test_acc))





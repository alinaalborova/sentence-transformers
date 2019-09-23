#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
import math
import logging
from datetime import datetime

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import joblib
import faiss
import numpy as np
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
from sentence_transformers.util import batch_to_device


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# ## Training

# In[2]:


# Read the dataset
batch_size = 16
nli_reader = NLIDataReader('datasets/AllNLI')
sts_reader = STSDataReader('datasets/stsbenchmark')
train_num_labels = nli_reader.get_num_labels()
model_save_path = 'output/training_nli_bert-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# In[3]:


# Use BERT for mapping tokens to embeddings
# Using manually downloaded model data:
word_embedding_model = models.BERT('../models/bert-base-multilingual-cased/')
# Or you can let the library handle the downloading and caching for you:
# word_embedding_model = models.BERT('bert-base-multilingual-cased')


# In[4]:


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


# In[5]:


set_trainable(word_embedding_model.bert.embeddings.word_embeddings, False)
print(word_embedding_model.bert.embeddings.word_embeddings.weight.requires_grad)
print(word_embedding_model.bert.embeddings.position_embeddings.weight.requires_grad)


# In[6]:


# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[7]:


# # Convert the dataset to a DataLoader ready for training
# logging.info("Read AllNLI train dataset")
# train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# train_loss = losses.SoftmaxLoss(
#     model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

# logging.info("Read STSbenchmark dev dataset")
# dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
# dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
# evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


# In[8]:


# joblib.dump(train_data, "allnli_train_dataset.jl")
# joblib.dump(dev_data, "sts_dev_dataset.jl")


# In[9]:


# Convert the dataset to a DataLoader ready for training
logging.info("Read AllNLI train dataset")
train_data = joblib.load("allnli_train_dataset.jl")
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(
    model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

logging.info("Read STSbenchmark dev dataset")
dev_data = joblib.load("sts_dev_dataset.jl")
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


# In[10]:


# Configure the training
num_epochs = 1

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )


# In[90]:


model_save_path


# In[11]:


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

model.evaluate(evaluator)


# ## Tatoeba

# In[15]:


from pathlib import Path
from sentence_transformers.readers.InputExample import InputExample


# In[26]:


class TatoebaReader:
    """Reads in a plain text file, in which every line contains one 
    sentence."""
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def get_examples(self):
        examples = []        
        with open(self.file_path) as fin:
            for i, line in enumerate(fin.readlines()):
                examples.append(InputExample(guid=i, texts=[line], label=0))                
        return examples


# In[33]:


TATOEBA_PATH = Path("../data/tatoeba/v1/")

tatoeba_engcmn_eng = TatoebaReader(TATOEBA_PATH / "tatoeba.cmn-eng.eng")
ds = SentencesDataset(tatoeba_engchm_eng.get_examples(), model=model)
tatoeba_engcmn_eng_loader = DataLoader(ds, shuffle=False, batch_size=32, collate_fn=model.smart_batching_collate)

tatoeba_engcmn_cmn = TatoebaReader(TATOEBA_PATH / "tatoeba.cmn-eng.cmn")
ds = SentencesDataset(tatoeba_engcmn_cmn.get_examples(), model=model)
tatoeba_engcmn_cmn_loader = DataLoader(ds, shuffle=False, batch_size=32, collate_fn=model.smart_batching_collate)


# In[75]:


get_ipython().run_cell_magic('time', '', 'cmn_embeddings, eng_embeddings = [], []\nwith torch.no_grad():\n    for batch in tatoeba_engcmn_cmn_loader:\n        cmn_embeddings.append(model(\n            batch_to_device(batch, "cuda")[0][0]\n        )[\'sentence_embedding\'])\n    for batch in tatoeba_engcmn_eng_loader:\n        eng_embeddings.append(model(\n            batch_to_device(batch, "cuda")[0][0]\n        )[\'sentence_embedding\'])\ncmn_embeddings = torch.cat(cmn_embeddings).cpu().numpy()\neng_embeddings = torch.cat(eng_embeddings).cpu().numpy()')


# In[76]:


len(eng_embeddings)


# In[79]:


cmn_idx = faiss.IndexFlatL2(cmn_embeddings.shape[1])
faiss.normalize_L2(cmn_embeddings)
cmn_idx.add(cmn_embeddings)
eng_idx = faiss.IndexFlatL2(eng_embeddings.shape[1])
faiss.normalize_L2(eng_embeddings)
eng_idx.add(eng_embeddings)


# In[82]:


_, eng_to_cmn = cmn_idx.search(x=eng_embeddings, k=1)
_, cmn_to_eng = eng_idx.search(x=cmn_embeddings, k=1)


# In[88]:


np.sum(eng_to_cmn[:, 0] == np.arange(1000))


# In[89]:


np.sum(cmn_to_eng[:, 0] == np.arange(1000))


# ## Baseline

# In[91]:


word_embedding_model = models.BERT('../models/bert-base-multilingual-cased/')
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[92]:


get_ipython().run_cell_magic('time', '', 'cmn_embeddings, eng_embeddings = [], []\nwith torch.no_grad():\n    for batch in tatoeba_engcmn_cmn_loader:\n        cmn_embeddings.append(model(\n            batch_to_device(batch, "cuda")[0][0]\n        )[\'sentence_embedding\'])\n    for batch in tatoeba_engcmn_eng_loader:\n        eng_embeddings.append(model(\n            batch_to_device(batch, "cuda")[0][0]\n        )[\'sentence_embedding\'])\ncmn_embeddings = torch.cat(cmn_embeddings).cpu().numpy()\neng_embeddings = torch.cat(eng_embeddings).cpu().numpy()')


# In[93]:


cmn_idx = faiss.IndexFlatL2(cmn_embeddings.shape[1])
faiss.normalize_L2(cmn_embeddings)
cmn_idx.add(cmn_embeddings)
eng_idx = faiss.IndexFlatL2(eng_embeddings.shape[1])
faiss.normalize_L2(eng_embeddings)
eng_idx.add(eng_embeddings)
_, eng_to_cmn = cmn_idx.search(x=eng_embeddings, k=1)
_, cmn_to_eng = eng_idx.search(x=cmn_embeddings, k=1)


# In[94]:


np.sum(eng_to_cmn[:, 0] == np.arange(1000))


# In[95]:


np.sum(cmn_to_eng[:, 0] == np.arange(1000))


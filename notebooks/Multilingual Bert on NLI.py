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
from pathlib import Path

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import joblib
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
from sentence_transformers.util import batch_to_device
from sentence_transformers.readers.InputExample import InputExample


# In[2]:


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

# In[2]:


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

TATOEBA_PATH = Path("../data/tatoeba/v1/")


# In[3]:


def evaluate_language_pair(model, pair_name="cmn-eng", batch_size=32):
    lang_1, lang_2 = pair_name.split("-")
    reader_1 = TatoebaReader(TATOEBA_PATH / f"tatoeba.{pair_name}.{lang_1}")
    ds_1 = SentencesDataset(reader_1.get_examples(), model=model)
    loader_1 = DataLoader(
        ds_1, shuffle=False, batch_size=batch_size, 
        collate_fn=model.smart_batching_collate)

    reader_2 = TatoebaReader(TATOEBA_PATH / f"tatoeba.{pair_name}.{lang_2}")
    ds_2 = SentencesDataset(reader_2.get_examples(), model=model)
    loader_2 = DataLoader(
        ds_2, shuffle=False, batch_size=batch_size, 
        collate_fn=model.smart_batching_collate)
    
    model.eval()
    emb_1, emb_2 = [], []
    with torch.no_grad():
        for batch in loader_1:
            emb_1.append(model(
                batch_to_device(batch, "cuda")[0][0]
            )['sentence_embedding'])
        for batch in loader_2:
            emb_2.append(model(
                batch_to_device(batch, "cuda")[0][0]
            )['sentence_embedding'])
    emb_1 = torch.cat(emb_1).cpu().numpy()
    emb_2 = torch.cat(emb_2).cpu().numpy()
    
    idx_1 = faiss.IndexFlatL2(emb_1.shape[1])
    faiss.normalize_L2(emb_1)
    idx_1.add(emb_1)
    idx_2 = faiss.IndexFlatL2(emb_2.shape[1])
    faiss.normalize_L2(emb_2)
    idx_2.add(emb_2)
    
    results = []
    _, match = idx_2.search(x=emb_1, k=1)
    results.append((
        lang_1, lang_2,
        np.sum(match[:, 0] == np.arange(len(emb_1))),
        len(emb_1)
    ))
    _, match = idx_1.search(x=emb_2, k=1)
    results.append((
        lang_2, lang_1,
        np.sum(match[:, 0] == np.arange(len(emb_2))),
        len(emb_2)
    ))
    return results


# In[4]:


PAIRS = ["ita-eng", "spa-eng", "fra-eng", "deu-eng", "rus-eng", "jpn-eng", "cmn-eng", "hin-eng"]


# ### Fine-tuned

# In[5]:


model = SentenceTransformer('output/training_nli_bert-2019-09-22_21-50-05')


# In[6]:


results = []
for pair in PAIRS:
    results += evaluate_language_pair(model, pair_name=pair, batch_size=50)
df_finetuned = pd.DataFrame(results, columns=["from", "to", "correct", "total"])


# In[7]:


df_finetuned


# ## Baseline

# In[8]:


word_embedding_model = models.BERT('../models/bert-base-multilingual-cased/')
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[9]:


results = []
for pair in PAIRS:
    results += evaluate_language_pair(model, pair_name=pair, batch_size=50)
df_baseline_mean = pd.DataFrame(results, columns=["from", "to", "correct", "total"])
df_baseline_mean


# In[10]:


word_embedding_model = models.BERT('../models/bert-base-multilingual-cased/')
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[11]:


results = []
for pair in PAIRS:
    results += evaluate_language_pair(model, pair_name=pair, batch_size=50)
df_baseline_cls = pd.DataFrame(results, columns=["from", "to", "correct", "total"])


# In[12]:


df_baseline_cls


# In[13]:


word_embedding_model = models.BERT('../models/bert-base-multilingual-cased/')
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=True)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[14]:


results = []
for pair in PAIRS:
    results += evaluate_language_pair(model, pair_name=pair, batch_size=50)
df_baseline_max = pd.DataFrame(results, columns=["from", "to", "correct", "total"])


# In[15]:


df_baseline_max


# ### Comparison

# In[16]:


df_baseline_mean["err_mean"] = 1 - df_baseline_mean["correct"] / df_baseline_mean["total"]
df_baseline_max["err_max"] = 1 - df_baseline_max["correct"] / df_baseline_max["total"]
df_baseline_cls["err_cls"] = 1 - df_baseline_cls["correct"] / df_baseline_cls["total"]
df_finetuned["err_finetuned"] = 1 - df_finetuned["correct"] / df_finetuned["total"]


# In[17]:


df_err = pd.concat([
    df.set_index(["from", "to"]).drop(["correct", "total"], axis=1)
    for df in (df_baseline_mean, df_baseline_max, df_baseline_cls, df_finetuned)
], axis=1)
df_err


# In[19]:


df_err["diff_mean"] = df_err["err_finetuned"] - df_err["err_mean"]
df_err["diff_pct_mean"] = df_err["diff_mean"] / df_err["err_mean"]
df_err["diff_max"] = df_err["err_finetuned"] - df_err["err_max"]
df_err["diff_pct_max"] = df_err["diff_max"] / df_err["err_max"]
df_err["diff_cls"] = df_err["err_finetuned"] - df_err["err_cls"]
df_err["diff_pct_cls"] = df_err["diff_cls"] / df_err["err_cls"]
df_err


# In[20]:


df_err.to_csv("df_err.csv")


# In[ ]:





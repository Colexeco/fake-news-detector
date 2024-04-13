#https://medium.com/@skillcate/detecting-fake-news-with-a-bert-model-9c666e3cdd9b
import polars as pl
import numpy as np
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import opendatasets as od
device = torch.device("cuda")
dataset = 'https://www.kaggle.com/datasets/jainpooja/fake-news-detection'
od.download(dataset)

true_news = pl.read_csv("./fake-news-detection/True.csv")
fake_news = pl.read_csv("./fake-news-detection/Fake.csv")
#Add true/false features to each article
true_news = true_news.with_columns(Target = pl.Series(['True']*len(true_news)))
fake_news = fake_news.with_columns(Target = pl.Series(['False']*len(fake_news)))
#Merge true and fake news into one dataset
news = pl.concat([true_news,fake_news])
#Change from true/false to 0/1
news = news.with_columns(news['Target'].apply(lambda x: 1 if x == 'True' else 0).alias('label'))
#Shuffle rows
news = news.with_columns(random = pl.Series(np.random.rand(len(news))))
news = news.sort("random")
news = news.drop("random")
label_size = [news['label'].sum(),len(news['label'])-news['label'].sum()]
#percentage of true and fake news
plt.pie(label_size,explode=[0.1,0.1],colors=['firebrick','navy'],startangle=90,shadow=True,labels=['Fake','True'],autopct='%1.1f%%')
plt.savefig('pie.png')
# Train-Validation-Test set split into 70:15:15 ratio
news = news.to_pandas()
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(news['title'], news['label'], random_state=2018, test_size=0.3, stratify=news['Target'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)
# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Plot histogram of the number of words in train data 'title'
seq_len = [len(title.split()) for title in train_text]
pl.Series(seq_len).hist(bins = 40)
plt.figure()
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')
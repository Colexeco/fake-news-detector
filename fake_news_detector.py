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
shuffled_news = news.sort("random")
shuffled_news = shuffled_news.drop("random")
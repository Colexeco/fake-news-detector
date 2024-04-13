#kaggle datasets download -d jainpooja/fake-news-detection
import numpy as np
import pandas as pd
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

true_news = pd.read_csv("./fake-news-detection/True.csv")
fake_news = pd.read_csv("./fake-news-detection/Fake.csv")


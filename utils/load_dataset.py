import os
import csv
import requests
import pandas as pd 
import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from os.path import join

def load_dataset(dataset_name):
    if dataset_name == 'TUANDROMD':
        CSV_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00622/TUANDROMD.csv'
        with requests.Session() as s:
            download = s.get(CSV_URL)

            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            df = pd.DataFrame(my_list[1:],columns=my_list[0])
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
            df.loc[(df.Label == 'malware'),'Label']= '1'
            df.loc[(df.Label == 'goodware'),'Label']= '0'
            labels = df.pop('Label').values
            X_train, X_test, Y_train, Y_test = train_test_split(df.astype(float), labels.astype(int), random_state=42)
            train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(Y_train)))
            test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(Y_test)).type(torch.FloatTensor))
            return train_dataset, test_dataset
    
    elif dataset_name == 'BlogFeedback':
        train_data_np = []
        test_data_np = []
        for file in os.listdir('./datasets/BlogFeedback'):
            if "test" in file: 
                with open('./datasets/BlogFeedback/' + file, newline='') as csvfile:
                    cr = csv.reader(csvfile, delimiter=',')
                    my_list_test = list(cr)
                    test_data_np.extend(my_list_test)
            if "train" in file: 
                with open('./datasets/BlogFeedback/' + file, newline='') as csvfile:
                    cr = csv.reader(csvfile, delimiter=',')
                    my_list_train = list(cr)
                    train_data_np.extend(my_list_train)

        train_df = pd.DataFrame(train_data_np)
        test_df = pd.DataFrame(test_data_np)
        train_labels = train_df.pop(280).values
        test_labels = test_df.pop(280).values

        scaler = MinMaxScaler()
        train_df[train_df.columns] = scaler.fit_transform(train_df[train_df.columns])
        test_df[test_df.columns] = scaler.fit_transform(test_df[test_df.columns])

        train_labels[train_labels.astype(float) > 0.0] = 1.0
        train_labels[train_labels.astype(float) == 0.0] = 0.0

        test_labels[test_labels.astype(float) > 0.0] = 1.0
        test_labels[test_labels.astype(float) == 0.0] = 0.0
        
        train_dataset = data.TensorDataset(torch.tensor(np.array(train_df)).type(torch.FloatTensor), torch.tensor(np.array(train_labels).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(test_df)).type(torch.FloatTensor), torch.tensor(np.array(test_labels).astype(int)))
        return train_dataset, test_dataset

    elif dataset_name == "BreastCancer":
        X, y = load_breast_cancer(return_X_y=True)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(Y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(Y_test))
       
        return train_dataset, test_dataset

    elif dataset_name == 'reuters':
        train_dataset = np.load('datasets/reuters/reutersidf10k_train.npy', allow_pickle=True)
        test_dataset = np.load('datasets/reuters/reutersidf10k_test.npy', allow_pickle=True)

        X_train = train_dataset.item()['data']
        Y_train = train_dataset.item()['label']
        X_test = test_dataset.item()['data']
        Y_test = test_dataset.item()['label']

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(Y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(Y_test))

        return train_dataset, test_dataset
        
        
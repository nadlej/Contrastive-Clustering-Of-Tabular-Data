import os
import csv
import requests
import pandas as pd 
import numpy as np
import torch
import torchvision
from modules import transform
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils import data
import datasets
from scipy.spatial.distance import cdist


def load_dataset(dataset_name):
    if dataset_name == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root="./datasets",
            download=True,
            train=True,
            transform=transform.Transforms(),
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./datasets",
            download=True,
            train=False,
            transform=transform.Transforms(),
        )
        
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        return dataset
    
    elif  dataset_name == 'TUANDROMD':
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
            dataset = data.ConcatDataset([train_dataset, test_dataset])
            return dataset  
    
    elif dataset_name == 'BlogFeedback':
        train_data_np = []
        test_data_np = []
        for file in os.listdir('../datasets/BlogFeedback'):
            if "test" in file: 
                with open('../datasets/BlogFeedback/' + file, newline='') as csvfile:
                    cr = csv.reader(csvfile, delimiter=',')
                    my_list_test = list(cr)
                    test_data_np.extend(my_list_test)
            if "train" in file: 
                with open('../datasets/BlogFeedback/' + file, newline='') as csvfile:
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

        scaler.fit(X)
        X = scaler.transform(X)

        dataset = data.TensorDataset(torch.tensor(X).type(torch.FloatTensor), torch.tensor(y))
       
        return dataset

    elif dataset_name == 'reuters':
        train_dataset = np.load('datasets/reuters/reutersidf10k_train.npy', allow_pickle=True)

        X = train_dataset.item()['data']
        y = train_dataset.item()['label']
        
        print(len(y))
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        dataset = data.TensorDataset(torch.tensor(X).type(torch.FloatTensor), torch.tensor(y))
        return dataset

    elif dataset_name == 'letter':
        LETTER_DATA = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        with requests.Session() as s:
            download = s.get(LETTER_DATA)

            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            df = pd.DataFrame(my_list)
            labels = df.pop(0).values
        
            scaler = MinMaxScaler()
            X = scaler.fit_transform(df)

            le = preprocessing.LabelEncoder()
            y = le.fit_transform(labels)
            
            dataset = data.TensorDataset(torch.tensor(np.array(X)).type(torch.FloatTensor), torch.tensor(np.array(y).astype(int)))
            return dataset
        
    elif dataset_name == 'ColorectalCarcinoma':
        dataset = datasets.load_dataset("wwydmanski/colorectal-carcinoma-microbiome-fengq", "presence-absence")
        train_dataset, test_dataset = dataset['train'], dataset['test']

        scaler = StandardScaler()
        X_train = np.array(train_dataset['values'])
        y_train = np.array(train_dataset['target'])
        X_train = scaler.fit_transform(X_train)

        X_test = np.array(test_dataset['values'])
        y_test = np.array(test_dataset['target'])
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(y_train).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(y_test).astype(int)))

        dataset = data.ConcatDataset([train_dataset, test_dataset])
        return dataset
    
    elif dataset_name == 'ColorectalCarcinomaCLR':
        dataset = datasets.load_dataset("wwydmanski/colorectal-carcinoma-microbiome-fengq", "CLR")
        train_dataset, test_dataset = dataset['train'], dataset['test']
        
        scaler = StandardScaler()
        X_train = np.array(train_dataset['values'])
        y_train = np.array(train_dataset['target'])
        X_train = scaler.fit_transform(X_train)

        X_test = np.array(test_dataset['values'])
        y_test = np.array(test_dataset['target'])
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(y_train).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(y_test).astype(int)))

        dataset = data.ConcatDataset([train_dataset, test_dataset])
        return dataset
    
    elif dataset_name == "Bioresponse":
        dataset = datasets.load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/Bioresponse.csv")
        df = pd.DataFrame(dataset['train'])
        y = df.pop('target').values
        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        dataset = data.ConcatDataset([train_dataset, test_dataset])
        return dataset
    elif dataset_name == "EyeMovements":
        dataset = datasets.load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/eye_movements.csv")
        df = pd.DataFrame(dataset['train'])
        y = df.pop('label').values
        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        dataset = data.ConcatDataset([train_dataset, test_dataset])
        return dataset
    else:
        raise NotImplementedError
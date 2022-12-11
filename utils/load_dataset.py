import os
import csv
import requests
import pandas as pd 
import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

    # elif dataset_name == 'reuters':
    #     data_path='./datasets/reuters'
    #     if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
    #         print('making reuters idf features')
    #         make_reuters_data(data_path)
    #         print('reutersidf saved to ' + data_path)
    #     data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    #     # has been shuffled
    #     x = data['data']
    #     y = data['label']
    #     x = x.reshape((x.shape[0], x.size / x.shape[0])).astype('float64')
    #     y = y.reshape((y.size,))
    #     print('REUTERSIDF10K samples', x.shape)
    #     return x, y
    
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

        print(type(train_df))
        print(type(train_labels))
        

        train_dataset = data.TensorDataset(torch.tensor(np.array(train_df)).type(torch.FloatTensor), torch.tensor(np.array(train_labels).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(test_df)).type(torch.FloatTensor), torch.tensor(np.array(test_labels).astype(int)))
        return train_dataset, test_dataset


def make_reuters_data(data_dir):
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in did_to_cat.keys():
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did_to_cat.has_key(did):
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000]
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], x.size / x.shape[0]))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})
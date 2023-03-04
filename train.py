import os
import numpy as np
import torch
import argparse
import torch.optim
import optuna
from modules import network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from matplotlib import pyplot as plt
from cluster import cluster
from utils.load_dataset import load_dataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from evaluation import evaluation
from utils.generate_noise import generate_noisy_xbar
from comet_ml import Experiment

experiment = Experiment(
    api_key="api_key",
    project_name="name",
    workspace="nadlej",
)

def print_samples(x_i, x_j):
    for i in x_i:
        pixels_org = i.reshape((28,28))
        plt.imshow(pixels_org, cmap='gray')
        plt.show()
        break
    for j in x_j:
        pixels_org_2 = j.reshape((28,28))
        plt.imshow(pixels_org_2, cmap='gray')
        plt.show()
        break

def print_baselines_results(train_dataset, test_dataset, class_num):

    kmeans = KMeans(n_clusters=class_num) 
    train_dataset_bs = train_dataset.data.flatten(start_dim=1)
    test_dataset_bs = test_dataset.data.flatten(start_dim=1)

    kmeans.fit(train_dataset_bs)
    y_pred = kmeans.predict(test_dataset_bs.data)
    nmi, ari, f, acc = evaluation.evaluate(np.array(test_dataset.targets), np.array(y_pred))
    print('k-means accuracy on raw: ' + str(acc))
    print('k-means nmi on raw: ' + str(nmi))

    gm = GaussianMixture(n_components=class_num, n_init=10)
    gm.fit(train_dataset_bs)
    y_pred = gm.predict(test_dataset_bs.data)
    nmi, ari, f, acc = evaluation.evaluate(np.array(test_dataset.targets), np.array(y_pred))
    print('gmm accuracy: ' + str(acc))
    print('gmm nmi on raw: ' + str(nmi))

def train(params):
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # prepare data
    train_dataset, test_dataset = load_dataset(args.dataset)

    dataset = data.ConcatDataset([train_dataset, test_dataset])
    class_num = args.class_num
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # initialize model
    model = network.Network(args.input_size, params, class_num)
    model = model.to('cpu') 
    # optimizer / loss
    optimizer = getattr(torch.optim, params["optimizer"])(model.parameters(), 
                                                        lr=params['learning_rate'], 
                                                        weight_decay=args.weight_decay, 
                                                        betas=(0.9,0.999),
                                                        eps=params['eps'])
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cpu")
    criterion_instance = contrastive_loss.InstanceLoss(params['batch_size'], args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)

    if args.baselines:
        print_baselines_results(train_dataset, test_dataset, class_num)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = 0
        for step, (x, _) in enumerate(data_loader):
            optimizer.zero_grad()
            x_i = x.clone()
            x_j = x.clone()
            x_i = generate_noisy_xbar(x_i, params['noise'], params['masking_ratio'])
            x_j = generate_noisy_xbar(x_j, params['noise'], params['masking_ratio'])
            x_i = x_i.to('cpu')
            x_j = x_j.to('cpu') 
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            loss_instance = criterion_instance(z_i, z_j)
            loss_cluster = criterion_cluster(c_i, c_j)
            loss = loss_instance + loss_cluster
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)

    #test
    acc = cluster(params)
    return acc

def objective(trial):

    params = {
              'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1e-0),
              'eps': trial.suggest_float('eps', 1e-8, 1e-4),
              'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
              'batch_size': trial.suggest_int('batch_size', 1, 513),
              'projection_size': trial.suggest_int('projection_size', 127, 513),
              'n_layers': trial.suggest_int('n_layers', 0, 5),
              '0_layer_size': trial.suggest_int('0_layer_size', 127, 513),
              '1_layer_size': trial.suggest_int('1_layer_size', 127, 513),
              '2_layer_size': trial.suggest_int('2_layer_size', 127, 513),
              '3_layer_size': trial.suggest_int('3_layer_size', 127, 513),
              'masking_ratio': trial.suggest_float('masking_ratio', 0.2, 0.3),
              'noise': trial.suggest_categorical('noise', ['swap_noise', 'gaussian', 'mixed', 'zero'])
              }
    
    accuracy = train(params)
    print(accuracy)

    return accuracy

if __name__ == "__main__":
    search_space = {'learning_rate': [1e-3],
                    'eps': [1e-7],
                    'optimizer': ['AdamW'],
                    'batch_size': [128],
                    'projection_size': [256],
                    'n_layers': [3],
                    '0_layer_size': [512],
                    '1_layer_size': [256],
                    '2_layer_size': [128],
                    '3_layer_size': [128],
                    'masking_ratio': [0.2],
                    'noise': ['swap_noise']
                    }
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective)
    best_trial = study.best_trial

    f = open("save/MNIST/best_parameters.txt", "a")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
        f.write("{}: {}\n".format(key, value))
    
    print('Best accuracy: {}'.format(study.best_value))
    f.write('Best accuracy: {}'.format(study.best_value))
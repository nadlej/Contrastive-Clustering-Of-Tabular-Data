import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import network, transform
from evaluation import evaluation
from torch.utils import data
from utils.load_dataset import load_dataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def latent_cluster(model, train_dataset, test_dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    X_train, _ = inference(train_loader, model, device, args.latent_cluster)
    X_test, y_test = inference(test_loader, model, device, args.latent_cluster)

    kmeans = KMeans(n_clusters=args.class_num) 
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_test)
    nmi, ari, f, acc_kmeans = evaluation.evaluate(np.array(y_test), np.array(y_pred))
    print('k-means accuracy: ' + str(acc_kmeans))

    gm = GaussianMixture(n_components=args.class_num, n_init=10)
    gm.fit(X_train)
    y_pred = gm.predict(X_test)
    nmi, ari, f, acc_gmm = evaluation.evaluate(np.array(y_test), np.array(y_pred))
    print('gmm accuracy: ' + acc_gmm)

    return nmi, ari, f, acc_kmeans

def contrastive_cluster(model, train_dataset, test_dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers
    )

    X, Y = inference(data_loader, model, device, args.latent_cluster) 
    nmi, ari, f, acc = evaluation.evaluate(Y, X)

    return nmi, ari, f, acc

def inference(loader, model, device, latent_cluster=False):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            if latent_cluster:
                c = model.forward_backbone(x)
            else:
                c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def cluster(params):
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms().test_transform,
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.Transforms().test_transform,
        )
    elif args.dataset == 'TUANDROMD':
        train_dataset, test_dataset = load_dataset(args.dataset)
    elif args.dataset == 'BlogFeedback':
        train_dataset, test_dataset = load_dataset(args.dataset)
    elif args.dataset == 'BreastCancer':
        train_dataset, test_dataset = load_dataset(args.dataset)
    elif args.dataset == 'reuters':
        train_dataset, test_dataset = load_dataset(args.dataset)
    elif args.dataset == 'letter':
        train_dataset, test_dataset = load_dataset(args.dataset)
    else:
        raise NotImplementedError

    class_num = args.class_num
    model = network.Network(args.input_size, params, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epochs))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    if args.latent_cluster:
         nmi, ari, f, acc = latent_cluster(model, train_dataset, test_dataset, args)
    else:
        nmi, ari, f, acc = contrastive_cluster(model, train_dataset, test_dataset, args)
        
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return acc

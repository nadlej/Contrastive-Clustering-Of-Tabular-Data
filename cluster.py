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

def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
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
    else:
        raise NotImplementedError
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    class_num = args.class_num
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    model = network.Network(args.input_size, params, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epochs))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return acc

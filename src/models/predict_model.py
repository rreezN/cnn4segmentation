import pickle

import click
import torch
from train_model import PARAMS
from torch.utils.data import DataLoader, Dataset
from dataloader import standardiseTransform
from model import TheAudioBotV3
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from dataloader import MyDataModule

from torchsummary import summary
# sys.path.insert(0, 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/')
# basedir = 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/'

def plotConf(predictions, test_labels):
    cm = confusion_matrix(predictions, test_labels)
    confusion = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
    labs=['Other', 'Music', 'Human Voice', 'Engine Sounds', 'Alarm']
    df_cm = pd.DataFrame(confusion, columns=labs, index=labs)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    f, ax = plt.subplots(figsize=(5, 5))
    # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(df_cm, cbar=False, annot=True,  square=True, fmt='.2f',
                annot_kws={'size': 10}, linewidth=.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # plt.title('Confusion matrix')
    plt.tight_layout()
    plt.show()


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
def evaluate(model_filepath):
    print("Evaluating model")
    # model = TheAudioBotV3.load_from_checkpoint(model_filepath)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    # trainer = Trainer(devices="auto",accelerator=PARAMS["accelerator"])

    # data = np.load("data/raw/training.npy")
    # labels = np.load("data/raw/training_labels.npy")

    # train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(data, labels, test_size=0.1,
    #                                                                               random_state=11, shuffle=True)

    # data_loader = MyDataModule(batch_dict=PARAMS["batch_dict"], device=PARAMS["accelerator"], 
    #                            train_data=train_data_temp, train_labels=train_labels_temp, 
    #                            test_data=test_data, test_labels=test_labels)
    
    # trainer.test(model, datamodule=data_loader)

    model = TheAudioBotV3.load_from_checkpoint(model_filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(summary(model, (1, 32, 96)))        

    
    data = np.load("data/raw/training.npy")
    labels = np.load("data/raw/training_labels.npy")

    train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(data, labels, test_size=0.1,
                                                                                  random_state=11, shuffle=True)

    train_data_temp = torch.unsqueeze(torch.tensor(train_data_temp, dtype=torch.float32), 1)
    test_data = torch.unsqueeze(torch.tensor(test_data, dtype=torch.float32), 1)
    train_labels_temp = torch.tensor(train_labels_temp).long()
    test_labels = torch.tensor(test_labels).long()
    
    mu = torch.mean(train_data_temp, axis=(0,1,3))
    std = torch.std(train_data_temp, axis=(0,1,3))

    test_data = standardiseTransform(test_data, mu, std)    
    data = dataset(test_data, test_labels)
    
    dataloader = DataLoader(data, batch_size=1)

    correct, total = 0, 0
    predictions = []
    for batch in tqdm(dataloader):
        x, y = batch

        preds = model(x.to(device))
        preds = preds.argmax(dim=-1)
        predictions += preds.tolist()

        correct += (preds == y.to(device)).sum().item()
        total += y.numel()
    
    print(f"Test set accuracy {correct / total}")

    plotConf(predictions, test_labels)


if __name__ == "__main__":
    evaluate()

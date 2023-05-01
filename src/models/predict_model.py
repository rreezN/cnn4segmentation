import click
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop
from dataloader import standardiseTransform
from model import UNerveV1
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import pandas as pd
import seaborn as sns

# sys.path.insert(0, 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/')
# basedir = 'C:/Users/kr_mo/OneDrive-DTU/DTU/Andet/Oticon/AudioBots/'


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


def evaluate_predictions(predictions, labels):
    predictions = torch.argmin(predictions, dim=1)
    # t = ((labels == 0) & (predictions == 0)).to(torch.float32)
    # f = ((labels == 0) & (predictions == 1)).to(torch.float32)
    # q = ((labels == 255) & (predictions == 0)).to(torch.float32)
    # p = ((labels == 255) & (predictions == 1)).to(torch.float32)
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(q[0].cpu().numpy(), cmap='Greens')
    # axes[1].imshow(t[0].cpu().numpy(), cmap='Reds')
    # axes[2].imshow(f[0].cpu().numpy(), cmap='Blues')
    # plt.show()
    true_nerve = torch.sum((labels == 0) & (predictions == 0)).item()
    true_background = torch.sum((labels == 255) & (predictions == 1)).item()
    false_nerve = torch.sum((labels == 0) & (predictions == 1)).item()
    false_background = torch.sum((labels == 255) & (predictions == 0)).item()
    return true_background, true_nerve, false_background, false_nerve


def confusion_matrix(true_background, true_nerve, false_background, false_nerve):
    labs = ["background", "nerve"]
    background_sum = true_background + false_background
    nerve_sum = true_nerve + false_nerve
    true_background = true_background / background_sum
    true_nerve = true_nerve / nerve_sum
    false_background = false_background / background_sum
    false_nerve = false_nerve / nerve_sum
    df = pd.DataFrame([[true_background, false_background], [false_nerve, true_nerve]], columns=labs, index=labs)
    df.index.name = 'True'
    df.columns.name = 'Predicted'
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2%")
    plt.show()

@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
def evaluate(model_filepath):
    print("Evaluating model")

    model = UNerveV1.load_from_checkpoint(model_filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    image_files = glob("data/raw/512x512/test_images/*.png")
    images = []
    for image_file in image_files:
        img = Image.open(image_file)
        images.append(np.array(CenterCrop(508)(img)))

    real_test_data = torch.unsqueeze(torch.tensor(np.array(images), dtype=torch.float32), 1)

    data = np.load("data/processed/508x508/training_data.npy")
    data = torch.unsqueeze(torch.tensor(data, dtype=torch.float32), 1)
    test_data = np.load("data/processed/508x508/test_data.npy")
    test_data = torch.unsqueeze(torch.tensor(test_data, dtype=torch.float32), 1)
    test_labels = np.load("data/processed/508x508/test_labels.npy")
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    mu = torch.mean(data, axis=(0,1,3))
    std = torch.std(data, axis=(0,1,3))

    real_test_data_ = standardiseTransform(real_test_data, mu, std, standardise="standardise")
    test_data_ = standardiseTransform(test_data, mu, std, standardise="standardise")

    real_preds = model(real_test_data_.to(device))

    true_background, true_nerve, false_background, false_nerve = 0, 0, 0, 0

    ds = dataset(test_data_, test_labels)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    for i, (data, labels) in enumerate(dl):
        data = data.to(device)
        labels = labels.to(device)
        preds = model(data)
        true_background_, true_nerve_, false_background_, false_nerve_ = evaluate_predictions(preds, labels)
        true_background += true_background_
        true_nerve += true_nerve_
        false_background += false_background_
        false_nerve += false_nerve_

    confusion_matrix(true_background, true_nerve, false_background, false_nerve)

    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    axes[0, 0].set_ylabel("Test Images")
    axes[1, 0].set_ylabel("Predicted Segmentations")
    for i in range(4):
        image_array = torch.argmax(real_preds[i*3].view(1, 2, 420, 420), 1)[0].detach().cpu().numpy()
        test_img = CenterCrop(420)(real_test_data[i*3][0]).numpy()
        axes[0, i].imshow(test_img, cmap="gray")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        # axes[1, i].imshow(test_img, cmap="gray")
        # axes[1, i].imshow(image_array, cmap="Wistia_r", alpha=0.2)
        axes[1, i].imshow(image_array, cmap="gray")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        # axes[1, i].imshow(real_preds[i][0].detach().cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    evaluate()

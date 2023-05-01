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

from torchsummary import summary
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

    test_data = torch.unsqueeze(torch.tensor(images, dtype=torch.float32), 1)

    data = np.load("data/processed/508x508/training_data.npy")

    data = torch.unsqueeze(torch.tensor(data, dtype=torch.float32), 1)
    
    mu = torch.mean(data, axis=(0,1,3))
    std = torch.std(data, axis=(0,1,3))

    test_data_ = standardiseTransform(test_data, mu, std)

    preds = model(test_data_.to(device))

    fig, axes = plt.subplots(5, 3, figsize=(10, 16))
    for i in range(5):
        image_array = torch.argmax(preds[i].view(1, 2, 420, 420), 1)[0].detach().cpu().numpy()
        test_img = CenterCrop(420)(test_data[i][0]).numpy()
        axes[i, 0].imshow(test_img, cmap="gray")
        axes[i, 1].imshow(test_img, cmap="gray")
        axes[i, 1].imshow(image_array, cmap="Wistia_r", alpha=0.2)
        axes[i, 2].imshow(preds[i][0].detach().cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    evaluate()

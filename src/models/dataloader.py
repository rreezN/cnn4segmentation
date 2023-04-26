from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import numpy as np


def standardiseTransform(data, mean, std):
    # Standardise data
    return (data - mean) / std


class dataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)


class MyDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_dict,
            device,
            data_size="132x132",
            train_data=None,
            train_labels=None,
            val_data=None,
            val_labels=None,
            test_data=None,
            test_labels=None
    ):
        super().__init__()
        # batch_dict contains the batch scheduling
        self.batch_dict = batch_dict
        self.current_batch_size = batch_dict[0]
        self.device = "cuda" if device == "gpu" else "cpu"
        self.data_size = data_size
        self.init_train_data = train_data
        self.init_train_labels = train_labels
        self.init_val_data = val_data
        self.init_val_labels = val_labels
        self.init_test_data = test_data
        self.init_test_labels = test_labels

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders

        train_data = np.load(f"data/processed/{self.data_size}/training_data.npy") if self.init_train_data is None else self.init_train_data
        train_labels = np.load(f"data/processed/{self.data_size}/training_labels.npy") if self.init_train_data is None else self.init_train_labels

        train_data = torch.unsqueeze(torch.tensor(train_data, dtype=torch.float32), 1)
        train_labels = torch.tensor(train_labels != 255).long()
        
        # Standardisation parameters
        mu = torch.mean(train_data)
        std = torch.std(train_data)

        # Standardise train data
        train_data = standardiseTransform(train_data, mu, std)
        self.train_data = dataset(train_data, train_labels)

        # Standardise val data
        val_data = np.load(f"data/processed/{self.data_size}/val_data.npy") if self.init_val_data is None else self.init_val_data
        val_labels = np.load(f"data/processed/{self.data_size}/val_labels.npy") if self.init_val_data is None else self.init_val_labels

        val_data = torch.unsqueeze(torch.tensor(val_data, dtype=torch.float32), 1)
        val_labels = torch.tensor(val_labels != 255).long()
        val_data = standardiseTransform(val_data, mu, std)
        self.val_data = dataset(val_data, val_labels.long())

        # Standardise test data
        test_data = np.load(f"data/processed/{self.data_size}/test_data.npy") if self.init_test_data is None else self.init_test_data
        test_labels = np.load(f"data/processed/{self.data_size}/test_labels.npy") if self.init_test_data is None else self.init_test_labels

        test_data = torch.unsqueeze(torch.tensor(test_data, dtype=torch.float32), 1)
        test_labels = torch.tensor(test_labels != 255).long()
        test_data = standardiseTransform(test_data, mu, std)
        self.test_data = dataset(test_data, test_labels.long())

    def train_dataloader(self):
        # Update current_batch_size based on epoch number
        arr = np.array(list(self.batch_dict.keys()))
        key = np.max(arr[arr <= self.trainer.current_epoch])
        self.current_batch_size = self.batch_dict[key]
        # Return DataLoader with current_batch_size
        return DataLoader(self.train_data, batch_size=self.current_batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.current_batch_size, num_workers=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.current_batch_size, num_workers=1, shuffle=False)

    # def getWeightMap(self, x, y):
    #     _, mask_matrix = readData(x, y, H=132, W=132)
    #     weights = torch.zeros_like(mask_matrix)
    #     for im in range(len(batch)):
    #         weights[im] = weight_map(mask_matrix[im], 10, 5, background_class=0)
    #     return weights


if __name__ == "__main__":
    print("Running dataloader.py as main file.")
    batch_dict = {0: 8,
                   4: 16,
                   8: 24,
                   14: 32,
                   20: 48,
                   28: 64,
                   36: 96,
                   48: 128}
    PARAMS = {"batch_dict": batch_dict, "accelerator": "gpu" if torch.cuda.is_available() else "cpu"}
    data_loader = MyDataModule(batch_dict=PARAMS["batch_dict"], device=PARAMS["accelerator"], data_size="132x132")
    data_loader.setup()
    print("Train data size:", len(data_loader.train_data))
    print("Val data size:", len(data_loader.val_data))
    print("Test data size:", len(data_loader.test_data))
    pass

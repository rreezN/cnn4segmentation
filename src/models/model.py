import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchsummary import summary
from focal_loss import FocalLoss
import torch.nn as nn
import torch
import random
import torchaudio
import torchprofile


# Load the model
focal_loss = FocalLoss(
    alpha=torch.tensor([.73, .48, .91, .92, .97]),
    gamma=2,
    reduction='mean'
)

device = "cuda" if torch.cuda.is_available() else "cpu"
focal_loss = focal_loss.to(device)


class SpectrogramAugmentation(LightningModule):
    def __init__(self, p_time_shift=0.2, p_freq_shift=0.2, p_time_stretch=0.2, p_noise=0.2):
        super().__init__()
        self.p_time_shift = p_time_shift
        self.p_freq_shift = p_freq_shift
        self.p_time_stretch = p_time_stretch
        self.p_noise = p_noise

    def forward(self, x):
        probs = [random.random() for _ in range(4)]
        if probs[0] < self.p_time_shift:
            x = self.time_shift(x)
        if probs[1] < self.p_freq_shift:
            x = self.freq_shift(x)
        if probs[2] < self.p_time_stretch:
            x = self.time_stretch(x)
        if probs[3] < self.p_noise:
            x = self.add_noise(x)
        x = x.to(device)
        return x

    def time_shift(self, x):
        shift = random.randint(-x.shape[-1] // 10, x.shape[-1] // 10)
        return torch.roll(x, shifts=shift, dims=-1)

    def freq_shift(self, x):
        shift = random.randint(-x.shape[-2] // 10, x.shape[-2] // 10)
        return torch.roll(x, shifts=shift, dims=-2)
        
    def time_stretch(self, x):
        stretch_factor = random.uniform(0.8, 1.2)
        orig_device = x.device
        x = x.unsqueeze(0).cpu()
        x = torchaudio.transforms.TimeStretch(hop_length=None, n_freq=x.shape[-2], fixed_rate=stretch_factor)(x).float()
        x = x.squeeze(0).to(orig_device)
        return x

    def add_noise(self, x):
        noise_factor = random.uniform(0.01, 0.1)
        noise = torch.randn_like(x) * noise_factor
        return x + noise


# ----------------------------
# Audio Classification Model
# ----------------------------
class TheAudioBotBase(LightningModule):
    def __init__(self,
                 lr: float = 1e-3,
                 optimizer: str = "adam",
                 loss_function: str = "cross_entropy"):
        super().__init__()
        self.save_hyperparameters()
        self.conv = None
        self.ap = None
        self.lin = None
        self.lr = lr
        self.optimizer_type = optimizer
        self.loss_function = loss_function

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        augmentation = SpectrogramAugmentation()
        x = augmentation.forward(x)
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y).item() / (len(y) * 1.0)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return acc
    
    def on_validation_epoch_end(self):
        # check if the validation accuracy exceeds your threshold
        if self.trainer.current_epoch >= 20 and self.trainer.logged_metrics['val_acc'] < 0.80:
            # if the accuracy is below 0.9 after 5 epochs, stop the training early
            self.log('early_stop', True)
            self.logger.experiment.finish()
            self.trainer.should_stop = True

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y).item() / (len(y) * 1.0)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return acc

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer type not implemented.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               min_lr=1e-6, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def loss_func(self, y_hat, y):
        if self.loss_function == "cross_entropy":
            return F.cross_entropy(y_hat, y)
        elif self.loss_function == "focal_loss":
            return focal_loss(y_hat, y)
        else:
            raise ValueError("Loss function not implemented.")


class TheAudioBotV1(TheAudioBotBase):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=5)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)


class TheAudioBotV2(TheAudioBotBase):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = torch.nn.Dropout2d(p=0.4)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1, self.drop1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = torch.nn.Dropout2d(p=0.3)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2, self.drop2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = torch.nn.Dropout2d(p=0.2)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3, self.drop3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout2d(p=0.2)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4, self.drop4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                 nn.ReLU(),
                                 nn.Linear(in_features=128, out_features=5))

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)


class TheAudioBotV3(TheAudioBotBase):
    def __init__(self,
                 lr: float = 1e-3,
                 optimizer: str = "adam",
                 loss_function: str = "cross_entropy",
                 activation_function: str = "ReLU",
                 dropout: float = 0.4):
        super().__init__(lr=lr, optimizer=optimizer, loss_function=loss_function)
        self.activation_function = activation_function
        self.dropout = dropout
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = self.activation_func()
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1, self.drop1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu2 = self.activation_func()
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2, self.drop2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = self.activation_func()
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3, self.drop3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = self.activation_func()
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4,  self.bn4, self.drop4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Sequential(nn.Linear(in_features=256, out_features=200),
                                 nn.ReLU(),
                                 nn.Linear(in_features=200, out_features=128),
                                 nn.ReLU(),
                                 nn.Linear(in_features=128, out_features=5))
        
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def activation_func(self):
        if self.activation_function == "ReLU":
            return nn.ReLU()
        elif self.activation_function == "LeakyReLU":
            return nn.LeakyReLU()
        elif self.activation_function == "ELU":
            return nn.ELU()
        else:
            raise ValueError("Activation function not implemented.")


class TheAudioBotMini(TheAudioBotBase):
    def __init__(self,
                 lr: float = 1e-3,
                 optimizer: str = "adam",
                 loss_function: str = "cross_entropy",
                 activation_function: str = "ReLU",
                 dropout: float = 0.1):
        super().__init__(lr=lr, optimizer=optimizer, loss_function=loss_function)
        self.activation_function = activation_function
        self.dropout = dropout

        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = self.activation_func()
        self.bn1 = nn.BatchNorm2d(8)
        self.drop1 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1, self.drop1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu2 = self.activation_func()
        self.bn2 = nn.BatchNorm2d(16)
        self.drop2 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2, self.drop2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu3 = self.activation_func()
        self.bn3 = nn.BatchNorm2d(32)
        self.drop3 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3, self.drop3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = self.activation_func()
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4,  self.bn4, self.drop4]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Sequential(nn.Linear(in_features=64, out_features=128),
                                 nn.ReLU(),
                                 nn.Linear(in_features=128, out_features=64),
                                 nn.ReLU(),
                                 nn.Linear(in_features=64, out_features=5))

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def activation_func(self):
        if self.activation_function == "ReLU":
            return nn.ReLU()
        elif self.activation_function == "LeakyReLU":
            return nn.LeakyReLU()
        elif self.activation_function == "ELU":
            return nn.ELU()
        else:
            raise ValueError("Activation function not implemented.")


class TheAudioBotMiniV2(TheAudioBotBase):
    def __init__(self,
                 lr: float = 1e-3,
                 optimizer: str = "adam",
                 loss_function: str = "cross_entropy",
                 activation_function: str = "ReLU",
                 dropout: float = 0.1):
        super().__init__(lr=lr, optimizer=optimizer, loss_function=loss_function)
        self.activation_function = activation_function
        self.dropout = dropout

        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = self.activation_func()
        self.bn1 = nn.BatchNorm2d(24)
        self.drop1 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1, self.drop1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = self.activation_func()
        self.bn2 = nn.BatchNorm2d(48)
        self.drop2 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2, self.drop2]

        # Third Convolution Block
        self.conv4 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = self.activation_func()
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = torch.nn.Dropout2d(p=self.dropout)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4,  self.bn4, self.drop4]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Sequential(nn.Linear(in_features=64, out_features=80),
                                 nn.ReLU(),
                                 nn.Linear(in_features=80, out_features=64),
                                 nn.ReLU(),
                                 nn.Linear(in_features=64, out_features=5))

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def activation_func(self):
        if self.activation_function == "ReLU":
            return nn.ReLU()
        elif self.activation_function == "LeakyReLU":
            return nn.LeakyReLU()
        elif self.activation_function == "ELU":
            return nn.ELU()
        else:
            raise ValueError("Activation function not implemented.")



if __name__ == "__main__":
    # Create the model and put it on the GPU if available
    myModel = TheAudioBotMiniV2()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)
    # Check that it is on Cuda
    next(myModel.parameters()).device
    print("Model Summary")
    summary(myModel, (1, 32, 96))
    inputs = torch.randn(1, 1, 32, 96)

    macs = torchprofile.profile_macs(myModel, inputs)
    print(f"GMACs: {macs/1e6}")

    # print('** Model architecture: ')
    # print(myModel)
    # print('\n** Learnable parameters in layers: ')
    # nr_parameters = 0
    # for p in myModel.parameters():
    #     print(p.shape)
    #     nr_parameters += p.numel()
    # print(f'\n** In total: {nr_parameters} parameters')

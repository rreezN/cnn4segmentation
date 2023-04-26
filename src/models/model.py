import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule
from torchsummary import summary
import torch.nn as nn
import torch
import random
import torchaudio
import torchvision
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

class NerveAugmentation(LightningModule):
    def __init__(self, p_blur=0.2, p_sharpen=0.2, p_noise=0.2):
        super().__init__()
        self.p_blur = p_blur
        self.p_sharpen = p_sharpen
        self.p_noise = p_noise

    def forward(self, x):
        probs = [random.random() for _ in range(3)]
        if probs[0] < self.p_blur:
            x = self.blur(x)
        if probs[1] < self.p_sharpen and probs[0] >= self.p_blur:
            x = self.sharpen(x)
        if probs[2] < self.p_noise:
            x = self.add_noise(x)
        x = x.to(device)
        return x

    @staticmethod
    def blur(x):
        """
        Function that applies a Gaussian blur to the input image
        :param x:
        :return:
        """
        blurrer = T.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2))
        return blurrer(x)

    @staticmethod
    def sharpen(x):
        """
        Function that applies a sharpening filter to the input image
        :param x:
        :return:
        """
        return T.functional.adjust_sharpness(x, sharpness_factor=random.uniform(1.5, 2))

    @staticmethod
    def add_noise(x):
        noise_factor = random.uniform(0.01, 0.1)
        noise = torch.randn_like(x) * noise_factor
        return x + noise


# ----------------------------
# Audio Classification Model
# ----------------------------
class CNN4AugBase(LightningModule):
    def __init__(self,
                 lr: float = 1e-3,
                 optimizer: str = "adam",
                 loss_function: str = "cross_entropy",
                 modelname: str = 'UNerveV1'):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer_type = optimizer
        self.loss_function = loss_function
        self.modelname= modelname
        self.x_to_plot = None
        self.y_to_plot = None

    def forward(self, x):
        if self.modelname == 'UNerveV1':
            # Encoder
            x1 = torch.relu(self.conv1_2(torch.relu(self.conv1_1(x))))
            x2 = torch.relu(self.conv2_2(torch.relu(self.conv2_1(self.pool(x1)))))
            x3 = torch.relu(self.conv3_2(torch.relu(self.conv3_1(self.pool(x2)))))
            x4 = torch.relu(self.conv4_2(torch.relu(self.conv4_1(self.pool(x3)))))
            x5 = torch.relu(self.conv5_2(torch.relu(self.conv5_1(self.pool(x4)))))

            # Decoder
            x6 = torch.relu(self.upconv1(x5))
            x4_ = torchvision.transforms.Resize(x6.shape[-2:], antialias=True)(x4)
            x6 = torch.cat((x4_, x6), dim=1)
            x6 = torch.relu(self.conv6_2(torch.relu(self.conv6_1(x6))))

            x7 = torch.relu(self.upconv2(x6))
            x3_ = torchvision.transforms.Resize(x7.shape[-2:], antialias=True)(x3)
            x7 = torch.cat((x3_, x7), dim=1)
            x7 = torch.relu(self.conv7_2(torch.relu(self.conv7_1(x7))))

            x8 = torch.relu(self.upconv3(x7))
            x2_ = torchvision.transforms.Resize(x8.shape[-2:], antialias=True)(x2)
            x8 = torch.cat((x2_, x8), dim=1)
            x8 = torch.relu(self.conv8_2(torch.relu(self.conv8_1(x8))))

            x9 = torch.relu(self.upconv4(x8))
            x1_ = torchvision.transforms.Resize(x9.shape[-2:], antialias=True)(x1)
            x9 = torch.cat((x1_, x9), dim=1)
            x9 = torch.relu(self.conv9_2(torch.relu(self.conv9_1(x9))))

            # Output
            out = self.out(x9)
            return out
        
        elif self.modelname == 'UNerveV2':
            x1 = torch.relu(self.conv1_2(torch.relu(self.conv1_1(x))))
            x2 = torch.relu(self.conv2_2(torch.relu(self.conv2_1(self.pool(x1)))))
            x3 = torch.relu(self.conv3_2(torch.relu(self.conv3_1(self.pool(x2)))))

            x4 = torch.relu(self.upconv1(x3))
            x2_ = torchvision.transforms.CenterCrop(x4.shape[-1])(x2)
            x4 = torch.cat((x2_, x4), dim=1)
            x4 = torch.relu(self.conv4_2(torch.relu(self.conv4_1(x4))))

            x5 = torch.relu(self.upconv2(x4))
            x1_ = torchvision.transforms.CenterCrop(x5.shape[-1])(x1)
            x5 = torch.cat((x1_, x5), dim=1)
            x5 = torch.relu(self.conv5_2(torch.relu(self.conv5_1(x5))))
            
            out = torch.softmax(self.out(x5), dim=1)
            
            return out
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        # augmentation = SpectrogramAugmentation()
        # x = augmentation.forward(x)
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # pred = torch.argmax(y_hat, dim=1)
        # acc = torch.sum(pred == y).item() / (len(y) * 1.0)
        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return acc
        self.x_to_plot = x[0]
        self.y_to_plot = y[0]
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        y_hat = self.forward(self.x_to_plot.view(1, 1, 132, 132))
        
        image_array = torch.argmax(y_hat, 1)[0].detach().cpu().numpy()
        # pred_image = wandb.Image(
        #     image_array, 
        #     caption=f"Prediction (Epoch: {self.trainer.current_epoch})"
        #     )
        
        label_array = self.y_to_plot.detach().cpu().numpy()
        # label_image = wandb.Image(
        #     label_array, 
        #     caption=f"Label (Epoch: {self.trainer.current_epoch})"
        #     )
        # wandb.log({"Predictions": image_array, "Labels": label_array"})

        # TODO: Det her virker, men tror Actual image med overlay er wrong... føler ikke det rigtige billede afspejler segmentation (se wandb).
        # TODO: Det er også lidt svært at se hvad der er hvad, så måske skal vi lave en anden måde at vise det på -> måske nogle andre cmaps el.lign.
        fig, ax = plt.subplots(1,3)
        fig.suptitle(f"Epoch: {self.trainer.current_epoch}", fontsize=16)
        ax[0].imshow(label_array)
        ax[0].set_title('Label')
        ax[1].imshow(image_array)
        ax[1].set_title('Prediction')
        ax[2].imshow(self.x_to_plot.view(132, 132).detach().cpu().numpy(), cmap='gray')
        ax[2].imshow(image_array, cmap="jet", alpha=0.5)
        ax[2].set_title('Actual Image\n with\n prediction overlay')
        plt.tight_layout()
        
        wandb.log({"Images": fig})
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.log('avg_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return {"avg_val_loss": avg_loss}

    # def on_validation_epoch_end(self):
    #     # check if the validation accuracy exceeds your threshold
    #     pass
    #     # if self.trainer.current_epoch >= 20 and self.trainer.logged_metrics['val_acc'] < 0.80:
    #     #     # if the accuracy is below 0.9 after 5 epochs, stop the training early
    #     #     self.log('early_stop', True)
    #     #     self.logger.experiment.finish()
    #     #     self.trainer.should_stop = True

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # pred = torch.argmax(y_hat, dim=1)
        # acc = torch.sum(pred == y).item() / (len(y) * 1.0)
        # self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return acc

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
        # elif self.loss_function == "focal_loss":
        #     return focal_loss(y_hat, y)
        else:
            raise ValueError("Loss function not implemented.")
    

class UNerveV1(CNN4AugBase):
    """
    Add skip layers should be added in this class. Add them for each up-sample block and add them to conv_layers.
    """
    def __init__(self, PARAMS):
        super().__init__(lr=PARAMS["learning_rate"], optimizer=PARAMS["optimizer"], loss_function=PARAMS["loss_function"], modelname=PARAMS['model_name'])
        self.learning_rate = PARAMS["learning_rate"]
        self.activation_function = PARAMS["activation_function"]
        self.dropout = PARAMS["dropout"]
        self.n_channels = PARAMS["n_channels"]
        self.n_classes = PARAMS["n_classes"]
        
        # Encoder
        self.conv1_1 = nn.Conv2d(1, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv1_1.weight, a=0.1)
        self.conv1_1.bias.data.zero_()
        self.conv1_2.bias.data.zero_()
        

        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        nn.init.kaiming_normal_(self.conv2_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2_2.weight, a=0.1)
        self.conv2_1.bias.data.zero_()
        self.conv2_2.bias.data.zero_()

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        nn.init.kaiming_normal_(self.conv3_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv3_2.weight, a=0.1)
        self.conv3_1.bias.data.zero_()
        self.conv3_2.bias.data.zero_()

        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        nn.init.kaiming_normal_(self.conv4_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv4_2.weight, a=0.1)
        self.conv4_1.bias.data.zero_()
        self.conv4_2.bias.data.zero_()

        self.conv5_1 = nn.Conv2d(512, 1024, 3)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3)
        nn.init.kaiming_normal_(self.conv5_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv5_2.weight, a=0.1)
        self.conv5_1.bias.data.zero_()
        self.conv5_2.bias.data.zero_()

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6_1 = nn.Conv2d(1024, 512, 3)
        self.conv6_2 = nn.Conv2d(512, 512, 3)
        nn.init.kaiming_normal_(self.conv6_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv6_2.weight, a=0.1)
        self.conv6_1.bias.data.zero_()
        self.conv6_2.bias.data.zero_()

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7_1 = nn.Conv2d(512, 256, 3)
        self.conv7_2 = nn.Conv2d(256, 256, 3)
        nn.init.kaiming_normal_(self.conv7_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv7_2.weight, a=0.1)
        self.conv7_1.bias.data.zero_()
        self.conv7_2.bias.data.zero_()

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8_1 = nn.Conv2d(256, 128, 3)
        self.conv8_2 = nn.Conv2d(128, 128, 3)
        nn.init.kaiming_normal_(self.conv8_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv8_2.weight, a=0.1)
        self.conv8_1.bias.data.zero_()
        self.conv8_2.bias.data.zero_()

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9_1 = nn.Conv2d(128, 64, 3)
        self.conv9_2 = nn.Conv2d(64, 64, 3)
        nn.init.kaiming_normal_(self.conv9_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv9_2.weight, a=0.1)
        self.conv9_1.bias.data.zero_()
        self.conv9_2.bias.data.zero_()

        # Output
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.out.weight, a=0.1)
        self.out.bias.data.zero_()


    def activation_func(self):
        if self.activation_function == "ReLU":
            return nn.ReLU()
        elif self.activation_function == "LeakyReLU":
            return nn.LeakyReLU()
        elif self.activation_function == "ELU":
            return nn.ELU()
        else:
            raise ValueError("Activation function not implemented.")


class UNerveV2(CNN4AugBase):
    """
    Add skip layers should be added in this class. Add them for each up-sample block and add them to conv_layers.
    """
    def __init__(self, PARAMS):
        super().__init__(lr=PARAMS["learning_rate"], optimizer=PARAMS["optimizer"], loss_function=PARAMS["loss_function"], modelname=PARAMS['model_name'])
        self.learning_rate = PARAMS["learning_rate"]
        self.activation_function = PARAMS["activation_function"]
        self.dropout = PARAMS["dropout"]
        self.n_channels = PARAMS["n_channels"]
        self.n_classes = PARAMS["n_classes"]
        
        # Encoder
        self.conv1_1 = nn.Conv2d(1, 8, 3)
        self.conv1_2 = nn.Conv2d(8, 8, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv1_1.weight, a=0.1)
        self.conv1_1.bias.data.zero_()
        self.conv1_2.bias.data.zero_()
        
        self.conv2_1 = nn.Conv2d(8, 16, 3)
        self.conv2_2 = nn.Conv2d(16, 16, 3)
        nn.init.kaiming_normal_(self.conv2_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv2_2.weight, a=0.1)
        self.conv2_1.bias.data.zero_()
        self.conv2_2.bias.data.zero_()

        self.conv3_1 = nn.Conv2d(16, 32, 3)
        self.conv3_2 = nn.Conv2d(32, 32, 3)
        nn.init.kaiming_normal_(self.conv3_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv3_2.weight, a=0.1)
        self.conv3_1.bias.data.zero_()
        self.conv3_2.bias.data.zero_()

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv4_1 = nn.Conv2d(32, 16, 3)
        self.conv4_2 = nn.Conv2d(16, 16, 3)
        nn.init.kaiming_normal_(self.conv4_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv4_2.weight, a=0.1)
        self.conv4_1.bias.data.zero_()
        self.conv4_2.bias.data.zero_()

        self.upconv2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.conv5_1 = nn.Conv2d(16, 8, 3)
        self.conv5_2 = nn.Conv2d(8, 8, 3)
        nn.init.kaiming_normal_(self.conv5_1.weight, a=0.1)
        nn.init.kaiming_normal_(self.conv5_2.weight, a=0.1)
        self.conv5_1.bias.data.zero_()
        self.conv5_2.bias.data.zero_()

        # Output
        self.out = nn.Conv2d(8, self.n_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.out.weight, a=0.1)
        self.out.bias.data.zero_()


    def activation_func(self):
        if self.activation_function == "ReLU":
            return nn.ReLU()
        elif self.activation_function == "LeakyReLU":
            return nn.LeakyReLU()
        elif self.activation_function == "ELU":
            return nn.ELU()
        else:
            raise ValueError("Activation function not implemented.")

# class UNerveV1(CNN4AugBase):
#     """
#     Add skip layers should be added in this class. Add them for each up-sample block and add them to conv_layers.
#     """
#     def __init__(self,
#                  lr: float = 1e-3,
#                  optimizer: str = "adam",
#                  loss_function: str = "cross_entropy",
#                  activation_function: str = "ReLU",
#                  dropout: float = 0.4):
#         super().__init__(lr=lr, optimizer=optimizer, loss_function=loss_function)
#         self.activation_function = activation_function
#         self.dropout = dropout
#         conv_layers = []

#         # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#         self.relu1 = self.activation_func()
#         self.bn1 = nn.BatchNorm2d(32)
#         self.drop1 = torch.nn.Dropout2d(p=self.dropout)
#         nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
#         self.conv1.bias.data.zero_()
#         conv_layers += [self.conv1, self.relu1, self.bn1, self.drop1]

#         # Second Convolution Block
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#         self.relu2 = self.activation_func()
#         self.bn2 = nn.BatchNorm2d(64)
#         self.drop2 = torch.nn.Dropout2d(p=self.dropout)
#         nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
#         self.conv2.bias.data.zero_()
#         conv_layers += [self.conv2, self.relu2, self.bn2, self.drop2]

#         # Third Convolution Block
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu3 = self.activation_func()
#         self.bn3 = nn.BatchNorm2d(128)
#         self.drop3 = torch.nn.Dropout2d(p=self.dropout)
#         nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
#         self.conv3.bias.data.zero_()
#         conv_layers += [self.conv3, self.relu3, self.bn3, self.drop3]

#         # Fourth Convolution Block
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu4 = self.activation_func()
#         self.bn4 = nn.BatchNorm2d(256)
#         self.drop4 = torch.nn.Dropout2d(p=self.dropout)
#         nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
#         self.conv4.bias.data.zero_()
#         conv_layers += [self.conv4, self.relu4,  self.bn4, self.drop4]

#         # Linear Classifier
#         self.ap = nn.AdaptiveAvgPool2d(output_size=1)
#         self.lin = nn.Sequential(nn.Linear(in_features=256, out_features=200),
#                                  nn.ReLU(),
#                                  nn.Linear(in_features=200, out_features=128),
#                                  nn.ReLU(),
#                                  nn.Linear(in_features=128, out_features=5))
        
#         # Wrap the Convolutional Blocks
#         self.conv = nn.Sequential(*conv_layers)

#     def activation_func(self):
#         if self.activation_function == "ReLU":
#             return nn.ReLU()
#         elif self.activation_function == "LeakyReLU":
#             return nn.LeakyReLU()
#         elif self.activation_function == "ELU":
#             return nn.ELU()
#         else:
#             raise ValueError("Activation function not implemented.")


if __name__ == "__main__":
    # Create the model and put it on the GPU if available
    myModel = UNerveV1()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)
    # Check that it is on Cuda
    next(myModel.parameters()).device
    print("Model Summary")
    summary(myModel, (1, 32, 96))
    inputs = torch.randn(1, 1, 32, 96)

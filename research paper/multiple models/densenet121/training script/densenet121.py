import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torchvision.models import densenet121
import os
from PIL import Image
import yaml
import matplotlib.pyplot as plt
from torchmetrics import Precision, Recall, F1Score


class YOLOv8Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt')
                                                    .replace('.jpeg', '.txt')
                                                    .replace('.png', '.txt'))

        # Read image
        image = Image.open(img_path).convert('RGB')

        # Read label
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    label = int(lines[0].split()[0])

        if self.transform:
            image = self.transform(image)

        return image, label


class PotholeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        with open(os.path.join(self.data_dir, 'data.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        train_img_dir = os.path.join(self.data_dir, 'train', 'images')
        train_label_dir = os.path.join(self.data_dir, 'train', 'labels')
        valid_img_dir = os.path.join(self.data_dir, 'valid', 'images')
        valid_label_dir = os.path.join(self.data_dir, 'valid', 'labels')

        self.train_dataset = YOLOv8Dataset(train_img_dir, train_label_dir, self.transform)
        self.val_dataset = YOLOv8Dataset(valid_img_dir, valid_label_dir, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


class PotholeModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Replace ResNet50 with DenseNet121
        self.model = densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)  # Single output for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')
        self.f1 = F1Score(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x).squeeze()
        preds = torch.sigmoid(y_hat) > 0.5

        # Update metrics
        precision = self.precision(preds, y.int())
        recall = self.recall(preds, y.int())
        f1 = self.f1(preds, y.int())

        # Log metrics
        self.log('val_precision', precision, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def plot_precision_recall_f1(trainer, save_path="precision_recall_f1.png"):
    """
    Plots Precision, Recall, and F1-Score over epochs.
    """
    # Extract logged metrics and convert tensors to Python scalars
    precision = float(trainer.callback_metrics.get("val_precision", 0.0))
    recall = float(trainer.callback_metrics.get("val_recall", 0.0))
    f1 = float(trainer.callback_metrics.get("val_f1", 0.0))

    # Generate plot
    epochs = range(1, trainer.current_epoch + 2)  # Current epoch is 0-indexed
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [precision] * len(epochs), label="Precision")
    plt.plot(epochs, [recall] * len(epochs), label="Recall")
    plt.plot(epochs, [f1] * len(epochs), label="F1-Score")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-Score Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main(data_dir, epochs=10, batch_size=16):
    data_module = PotholeDataModule(data_dir, batch_size)
    model = PotholeModel()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='pothole-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        devices=1
    )

    trainer.fit(model, data_module)

    # Save the trained model
    torch.save(model.state_dict(), "pothole_model_densenet.pth")

    # Save Precision, Recall, and F1-Score graph
    plot_precision_recall_f1(trainer, save_path="precision_recall_f1.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size)

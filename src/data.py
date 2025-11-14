from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
from PIL import Image
import logging
import warnings
import random

logger = logging.getLogger(__name__)

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor

class FoodDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Логика изменена для torchvision.datasets.Food101, 
        # который возвращает (PIL.Image, label)
        img, label = self.dataset[idx]
        img = img.convert("RGB") # Гарантируем, что изображение в RGB

        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_transforms(cfg, split="train"):
    if split == "train":
        transform_list = [
            T.Resize((cfg.transforms.common.resize, cfg.transforms.common.resize)),
            T.RandomHorizontalFlip(p=cfg.transforms.train.horizontal_flip_p),
            T.RandomRotation(cfg.transforms.train.random_rotation_deg),
            T.RandomResizedCrop(
                cfg.transforms.common.resize,
                scale=tuple(cfg.transforms.train.random_crop_scale)
            ),
            T.ColorJitter(
                brightness=cfg.transforms.train.color_jitter.brightness,
                contrast=cfg.transforms.train.color_jitter.contrast,
                saturation=cfg.transforms.train.color_jitter.saturation,
            ),
            T.ToTensor(),
            GaussianNoise(
                mean=cfg.transforms.train.gaussian_noise.mean,
                std=cfg.transforms.train.gaussian_noise.std,
            ),
            T.Normalize(mean=cfg.transforms.common.mean, std=cfg.transforms.common.std),
        ]
    else:  # val/test
        transform_list = [
            T.Resize((cfg.transforms.common.resize, cfg.transforms.common.resize)),
            T.ToTensor(),
            T.Normalize(mean=cfg.transforms.common.mean, std=cfg.transforms.common.std),
        ]
    return T.Compose(transform_list)

def get_loaders(cfg):
    """
    Создает и возвращает DataLoader'ы для обучающего и валидационного наборов данных.
    Работает с локально скачанным датасетом через torchvision.
    """
    # Загружаем датасет из локальной папки, указанной в конфиге
    import torchvision
    
    # Убедимся, что путь указан в конфиге
    if not hasattr(cfg.dataset, 'path'):
        raise ValueError("В конфиге (configs/base.yaml) должен быть указан путь к датасету: dataset.path")

    train_dataset_tv = torchvision.datasets.Food101(
        root=cfg.dataset.path, 
        split='train', 
        download=False # Важно: отключаем скачивание, т.к. данные уже локально
    )
    val_dataset_tv = torchvision.datasets.Food101(
        root=cfg.dataset.path, 
        split='test', 
        download=False
    )

    # Получаем трансформации
    train_transforms = get_transforms(cfg, "train")
    val_transforms = get_transforms(cfg, "val")

    # Создаем обертки FoodDataset
    train_dataset = FoodDataset(train_dataset_tv, transform=train_transforms)
    val_dataset = FoodDataset(val_dataset_tv, transform=val_transforms)

    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle, # Changed from hardcoded True
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        # generator=torch.Generator().manual_seed(cfg.train.seed) if cfg.train.manual_seed_enabled else torch.default_generator
        # Убрано для Windows из-за проблем с сериализацией при spawn; используем глобальный seed в train.py
        # Для Linux можно вернуть и добавить worker_init_fn для полной воспроизводимости
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.val.shuffle,
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        # generator=torch.Generator().manual_seed(cfg.train.seed) if cfg.train.manual_seed_enabled else torch.default_generator
        # Убрано для Windows из-за проблем с сериализацией при spawn; используй глобальный seed в train.py
        # Для Linux можно вернуть и добавить worker_init_fn для полной воспроизводимости
    )
    logger.info(f"Data loaders created with batch_size={cfg.train.batch_size}, num_workers={cfg.hardware.num_workers}")
    return train_loader, val_loader
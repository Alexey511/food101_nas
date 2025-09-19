import torch
import random
import numpy as np
from src.data import get_loaders
from src.models import get_model
from src.trainer import train_loop
from src.visualization import generate_plots
from omegaconf import OmegaConf
import logging
import datetime
import os

if __name__ == "__main__":
    base_cfg = OmegaConf.load("configs/base.yaml")
    #model_cfg = OmegaConf.load("configs/custom_tiny_vit.yaml")
    model_cfg = OmegaConf.load("configs/custom_cnn.yaml")
    #model_cfg = OmegaConf.load("configs/convnext_tiny_cnn.yaml")
    #model_cfg = OmegaConf.load("configs/resnet50_cnn.yaml")
    #model_cfg = OmegaConf.load("configs/timm_vit.yaml")
    cfg = OmegaConf.merge(base_cfg, model_cfg)

    # Фиксация seed для воспроизводимости, если включено
    if cfg.train.manual_seed_enabled:
        SEED = cfg.train.seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    # Создаём папку logs
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Генерация имени для логов и чекпоинтов
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    model_name_for_log = "unknown_model"
    if cfg.model.model_type == 'timm':
        model_name_for_log = cfg.model.timm_model_name
    elif cfg.model.model_type == 'custom':
        model_name_for_log = f"custom_{cfg.model.custom_model_type}"

    log_file = f"log_{timestamp}_{model_name_for_log}_batch{cfg.train.batch_size}.log"
    log_path = os.path.join(log_dir, log_file)

    # Создаём папку checkpoints с подпапкой для текущего запуска
    checkpoint_dir = os.path.join("checkpoints", f"{timestamp}_{model_name_for_log}_batch{cfg.train.batch_size}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Сохранение конфига
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    # Настройка логирования
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)
    logger.info("Starting training experiment with config saved at: %s", config_path)

    # Создание DataLoader
    train_loader, val_loader = get_loaders(cfg)
    model = get_model(cfg)
    model.to(cfg.hardware.device)
    _, _ = train_loop(model, train_loader, val_loader, cfg, checkpoint_dir) # Capture both return values

    # Построение графиков после обучения
    generate_plots(checkpoint_dir, cfg, logger)

    logger.info("Main training script completed")
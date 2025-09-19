import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from src.utils import cutmix
import os
from tqdm import tqdm
import logging
import numpy as np
import optuna
from omegaconf import OmegaConf
from typing import Literal, cast
import time
from src.utils import compute_custom_score

logger = logging.getLogger(__name__)

def get_model_name_for_log(cfg):
    """Возвращает имя модели для логирования и именования папок."""
    model_type = cfg.model.model_type
    if model_type == 'timm':
        return cfg.model.timm_model_name
    elif model_type == 'custom':
        return f"custom_{cfg.model.custom_model_type}"
    return "unknown_model"

def train_loop(model, train_loader, val_loader, cfg, checkpoint_dir, trial=None): # Добавлен аргумент trial=None
    # Определяем тип модели
    model_type = cfg.model.model_type
    is_custom_model = model_type == 'custom'
    
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(cfg.train, 'label_smoothing', 0.0))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    
    scheduler_name = cfg.train.scheduler.name
    scheduler_config = cfg.train.scheduler # Use scheduler_config to access all scheduler parameters

    if scheduler_name == 'cosine':
        t_max = getattr(scheduler_config, 't_max', cfg.train.epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name == 'onecycle':
        total_steps = len(train_loader) * cfg.train.epochs
        scheduler = OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=total_steps)
    elif scheduler_name == 'cosine_warm_restarts':
        t_0 = getattr(scheduler_config, 't_0', cfg.train.epochs) # Default T_0 to epochs
        t_mult = getattr(scheduler_config, 't_mult', 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult)
    elif scheduler_name == 'reduce_on_plateau':
        raw_mode = getattr(scheduler_config, 'mode', 'max')
        mode: Literal['min', 'max'] = 'max' # Default to 'max'
        if raw_mode in ['min', 'max']:
            mode = cast(Literal['min', 'max'], raw_mode) # Explicitly cast
        else:
            logger.warning(f"Invalid mode for ReduceLROnPlateau: {raw_mode}. Defaulting to 'max'.")
            mode = 'max' # Default if invalid
        factor = getattr(scheduler_config, 'factor', 0.1)
        patience = getattr(scheduler_config, 'patience', 10)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}. No scheduler will be used.")
        scheduler = None
    
    best_val_acc = 0.0
    metrics_file = os.path.join(checkpoint_dir, "metrics.txt")
    total_training_time = 0.0 # Initialize total training time
    
    logger.info(f"Starting training for {get_model_name_for_log(cfg)} with {cfg.train.epochs} epochs, lr={cfg.train.lr}, batch_size={cfg.train.batch_size}")
    logger.info(f"Model type: {model_type}, Custom model: {is_custom_model}")
    
    with open(metrics_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(cfg.train.epochs):
        epoch_start_time = time.time() # Start time for the current epoch
        # Train
        model.train() # Всегда вызываем model.train()
        
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_data_cutmix = [] # Для CutMix данных
        train_data_normal = [] # Для обычных данных (true, predicted)
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}"):
            images, labels = images.to(cfg.hardware.device), labels.to(cfg.hardware.device)

            # Инициализация для обычного случая
            inputs = images # Use original images by default
            labels_for_loss_a, labels_for_loss_b, lambda_cutmix = labels, labels, 1.0
            do_cutmix = cfg.train.cutmix
            
            # Проверка, включен ли CutMix, и применение, если да
            if do_cutmix:
                inputs, (labels_for_loss_a, labels_for_loss_b, lambda_cutmix) = cutmix(images, labels, alpha=1.0, prob=0.5)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if do_cutmix:
                loss = lambda_cutmix * criterion(outputs, labels_for_loss_a) + (1 - lambda_cutmix) * criterion(outputs, labels_for_loss_b)
            else:
                loss = criterion(outputs, labels) # Original labels when no cutmix
                
            loss.backward()
            
            # Gradient clipping если указано в конфиге
            if hasattr(cfg.train, 'gradient_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            # Обновление train_total и train_correct в зависимости от do_cutmix
            if do_cutmix:
                # For accuracy in cutmix, we usually compare with labels_a or the original labels
                # For simplicity and consistency with previous logic (which was flawed for accuracy),
                # let's assume accuracy is calculated against labels_a
                train_total += labels_for_loss_a.size(0)
                train_correct += (predicted == labels_for_loss_a).sum().item()
                # Собираем данные для train с CutMix
                train_data_cutmix.append(torch.stack([labels_for_loss_a, labels_for_loss_b, torch.full_like(labels_for_loss_a, lambda_cutmix), predicted], dim=1).cpu())
            else:
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                # Собираем данные для train без CutMix
                train_data_normal.append(torch.stack([labels, predicted], dim=1).cpu())
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Сохранение лейблов для train
        train_labels_dir = os.path.join(checkpoint_dir, "labels")
        os.makedirs(train_labels_dir, exist_ok=True)

        if do_cutmix and train_data_cutmix:
            train_data_array = torch.cat(train_data_cutmix, dim=0).numpy()
            train_labels_path = os.path.join(train_labels_dir, f"epoch_{epoch+1}_train_cutmix.npy")
            np.save(train_labels_path, train_data_array)
            logger.info(f"Saved train labels (CutMix) at {train_labels_path}")
        elif train_data_normal: # Если cutmix не применялся, но есть обычные данные
            train_data_array = torch.cat(train_data_normal, dim=0).numpy()
            train_labels_path = os.path.join(train_labels_dir, f"epoch_{epoch+1}_train_normal.npy")
            np.save(train_labels_path, train_data_array)
            logger.info(f"Saved train labels (Normal) at {train_labels_path}")
        
        # Validation
        val_loss, val_acc, val_labels, val_predicted, current_score = evaluate(model, val_loader, criterion, cfg, trial)
        
        # Calculate epoch time and add to total
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration
        logger.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")

        # Сохранение лейблов для val
        val_labels_dir = os.path.join(checkpoint_dir, "labels")
        os.makedirs(val_labels_dir, exist_ok=True)
        val_data_array = np.stack([val_labels, val_predicted], axis=1)
        val_labels_path = os.path.join(val_labels_dir, f"epoch_{epoch+1}_val.npy")
        np.save(val_labels_path, val_data_array)
        logger.info(f"Saved validation labels and predictions at {val_labels_path}")

        # Отчет для Optuna
        if trial is not None:
            trial.report(current_score, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                raise optuna.TrialPruned()
        
        # Обновление learning rate
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Логирование и сохранение метрик
        logger.info(f"Epoch {epoch+1}/{cfg.train.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Current Score: {current_score:.4f}")
        print(f"Epoch {epoch+1}/{cfg.train.epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        # Сохранение лучшей модели
        if cfg.train.save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, f"best_{get_model_name_for_log(cfg)}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Acc: {best_val_acc:.4f} at {best_model_path}")
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")
    
    # Сохранение последней модели
    last_model_path = os.path.join(checkpoint_dir, f"last_{get_model_name_for_log(cfg)}.pth")
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"Saved last model at {last_model_path}")
    avg_time_per_epoch = total_training_time / cfg.train.epochs
    logger.info(f"Training completed. Average time per epoch: {avg_time_per_epoch:.2f} seconds.")
    return best_val_acc, avg_time_per_epoch # Return average time per epoch

def evaluate(model, loader, criterion, cfg, trial=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    val_labels_true = []
    val_labels_predicted = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(cfg.hardware.device), labels.to(cfg.hardware.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_labels_true.append(labels.cpu())
            val_labels_predicted.append(predicted.cpu())
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total

    val_true_np = torch.cat(val_labels_true).numpy()
    val_predicted_np = torch.cat(val_labels_predicted).numpy()

    current_score = accuracy
    if trial:
        num_params = trial.user_attrs.get('num_params', 0)
        time_per_epoch = trial.user_attrs.get('avg_time_per_epoch', 0)
        current_score = compute_custom_score(accuracy, num_params, time_per_epoch, cfg)

    return avg_loss, accuracy, val_true_np, val_predicted_np, current_score
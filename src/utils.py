# src/utils.py
import torch
import numpy as np
import random
import logging

class Denormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def forward(self, tensor):
        return tensor * self.std + self.mean

def cutmix(data, target, alpha=1.0, prob=0.5):
    if torch.rand(1) > prob:
        return data, (target, target, 1.0)
    
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    batch_size = data.size(0)
    w, h = data.size(2), data.size(3)

    cut_rat = torch.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = torch.randint(0, w, (1,)).item()
    cy = torch.randint(0, h, (1,)).item()

    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)

    data[:, :, y1:y2, x1:x2] = shuffled_data[:, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

    return data, (target, shuffled_target, lam)

def compute_custom_score(acc, num_params, time_per_epoch, cfg):
    """
    Вычисляет кастомный score для оптимизации.
    - Применяет слабый линейный штраф до достижения целевых значений.
    - Применяет сильный квадратичный штраф после превышения целевых значений.
    """
    logger = logging.getLogger(__name__)
    if not hasattr(cfg.nas, 'custom_score'):
        logger.warning("Конфиг NAS не содержит 'nas.custom_score'. Используется только accuracy.")
        return acc

    weights = cfg.nas.custom_score.weights
    targets = cfg.nas.custom_score.targets

    # Комбинированный штраф за количество параметров
    params_ratio = num_params / targets['num_params']
    if params_ratio <= 1:
        params_penalty = params_ratio  # Линейный штраф в пределах бюджета
    else:
        params_penalty = params_ratio**2 # Квадратичный штраф при превышении

    # Комбинированный штраф за время на эпоху
    time_ratio = time_per_epoch / targets['time_per_epoch']
    if time_ratio <= 1:
        time_penalty = time_ratio  # Линейный штраф в пределах бюджета
    else:
        time_penalty = time_ratio**2 # Квадратичный штраф при превышении

    score = (
        weights['acc'] * acc +
        weights['num_params'] * params_penalty +
        weights['time_per_epoch'] * time_penalty
    )
    
    # Логгирование компонентов score для отладки
    logger.info(
        f"Trial Score Breakdown: "
        f"acc_term={weights['acc'] * acc:.4f}, "
        f"params_term={weights['num_params'] * params_penalty:.4f} (ratio={params_ratio:.2f}), "
        f"time_term={weights['time_per_epoch'] * time_penalty:.4f} (ratio={time_ratio:.2f})"
    )
    
    return score
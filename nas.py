#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) для Food101
Запуск поиска оптимальной архитектуры и гиперпараметров с помощью Optuna (поддержка ViT, CNN и других)
"""

import torch
import random
import numpy as np
import optuna
from src.data import get_loaders
from src.models import get_model, ViT, CustomCNN
from src.trainer import train_loop
from src.visualization import generate_plots
from omegaconf import OmegaConf
import logging
import datetime
import os
import argparse
import time
from src.utils import compute_custom_score
import shutil
from optuna.study import Study
from optuna.trial import FrozenTrial


def save_best_config_callback(study: Study, trial: FrozenTrial):
    """Callback to save the best model's config after each trial."""
    try:
        # Пытаемся получить лучший trial. Это может вызвать ValueError, если нет завершенных trial.
        best_trial = study.best_trial
        
        # Сохраняем конфиг, только если текущий trial — лучший
        if best_trial is not None and trial.number == best_trial.number:
            logger = logging.getLogger(__name__)
            logger.info(f"Trial {trial.number} is the new best trial so far with score {trial.value:.4f}.")
            
            # study.user_attrs теперь устанавливаются в main()
            experiment_dir = study.user_attrs.get("experiment_dir")
            # Конфиг теперь берем напрямую из user_attrs текущего trial
            best_config = trial.user_attrs.get("config")

            if not experiment_dir or not best_config:
                logger.warning("Could not save best config: 'experiment_dir' or 'config' not found in user_attrs.")
                return

            destination_path = os.path.join(experiment_dir, "best_overall_config.yaml")
            
            try:
                # Преобразуем dict обратно в OmegaConf для красивого сохранения
                conf_to_save = OmegaConf.create(best_config)
                OmegaConf.save(config=conf_to_save, f=destination_path)
                logger.info(f"Saved new best config from trial {trial.number} to {destination_path}")
            except Exception as e:
                logger.error(f"Failed to save best config to {destination_path}: {e}")

    except ValueError:
        # Это происходит, когда еще нет успешно завершенных trials (например, все были pruned)
        logger = logging.getLogger(__name__)
        logger.info(f"Callback for trial {trial.number} skipped: no completed trials yet to determine best trial.")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An unexpected error occurred in save_best_config_callback for trial {trial.number}: {e}")


def suggest_params(trial, params_cfg, prefix=""):
    """
    Универсальная функция для генерации параметров с помощью Optuna.
    """
    params = {}
    try:
        for param_name, param_info in params_cfg.items():
            p_type = param_info.get('type')
            full_name = f"{prefix}{param_name}" if prefix else param_name
            if p_type == 'int':
                params[param_name] = trial.suggest_int(full_name, param_info['range'][0], param_info['range'][1])
            elif p_type == 'float':
                params[param_name] = trial.suggest_float(full_name, param_info['range'][0], param_info['range'][1], log=param_info.get('log', False))
            elif p_type == 'categorical':
                params[param_name] = trial.suggest_categorical(full_name, param_info['options'])
            elif p_type.startswith('list_'):
                sub_type = p_type.split('_')[1]
                num_items = param_info['num_blocks']
                range_or_options = param_info.get('range_per_block') or param_info.get('options_per_block', [])
                params[param_name] = []
                for i in range(num_items):
                    item_name = f"{full_name}_{i}"
                    if sub_type == 'int':
                        val = trial.suggest_int(item_name, range_or_options[0], range_or_options[1])
                        params[param_name].append(val)
                    elif sub_type == 'float':
                        val = trial.suggest_float(item_name, range_or_options[0], range_or_options[1], log=param_info.get('log', False))
                        params[param_name].append(val)
                    elif sub_type == 'categorical':
                        val = trial.suggest_categorical(item_name, range_or_options)
                        params[param_name].append(val)
                    else:
                        raise ValueError(f"Неподдерживаемый подтип '{sub_type}' для list параметра '{param_name}'")
            else:
                raise ValueError(f"Неподдерживаемый тип параметра: {p_type} для {param_name}")
    except KeyError as e:
        logging.getLogger(__name__).error(f"Ошибка в конфиге параметров: отсутствует ключ {e}")
        raise
    return params

def objective_architecture_search(trial, cfg):
    """
    Objective функция для подбора параметров архитектуры.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Начало Trial {trial.number} для поиска архитектуры.")

    # Валидация конфига
    if not hasattr(cfg.nas.architecture_search, 'model_params'):
        raise ValueError("Конфиг NAS должен содержать 'nas.architecture_search.model_params'.")

    # Генерируем архитектуру
    arch_params = suggest_params(trial, cfg.nas.architecture_search.model_params, prefix="arch_")

    # Определяем тип модели
    model_type = cfg.model.custom_model_type if cfg.model.model_type == 'custom' else cfg.model.model_type

    # Специфика для ViT: коррекция dim и расчет mlp_dim
    if model_type == 'vit':
        # Проверка наличия необходимых ключей
        if 'heads' not in arch_params or 'dim' not in arch_params or 'mlp_ratio' not in arch_params:
             raise ValueError("Для ViT в model_params должны быть 'heads', 'dim' и 'mlp_ratio'.")

        desired_heads = arch_params['heads']
        desired_dim = arch_params['dim']
        
        # Находим совместимый dim
        dim = ViT.find_compatible_dim(desired_dim, desired_heads, config=cfg)
        arch_params['dim'] = dim
        
        # Рассчитываем mlp_dim на основе mlp_ratio
        arch_params['mlp_dim'] = int(dim * arch_params['mlp_ratio'])
        # Удаляем mlp_ratio, так как он не является параметром конструктора ViT
        del arch_params['mlp_ratio']

    print(f"\n   Конфигурация для Trial {trial.number} (архитектура, {model_type}):")
    for k, v in arch_params.items():
        print(f"      {k.capitalize()}: {v}")

    # Обновляем конфиг для создания модели
    for k, v in arch_params.items():
        setattr(cfg.model, k, v)
    
    # Явно проверяем наличие обязательных параметров
    try:
        _ = cfg.train.lr
        _ = cfg.model.dropout
        _ = cfg.train.batch_size
    except Exception as e:
        logger.error(f"Отсутствуют обязательные параметры (lr, dropout, batch_size) в конфиге для этапа поиска архитектуры: {e}")
        raise

    # Удаляем жестко заданные параметры, теперь они должны быть в конфиге
    cfg.train.epochs = cfg.nas.architecture_search.epochs_per_trial

    # Инициализация модели
    try:
        model = get_model(cfg)
        model.to(cfg.hardware.device)
    except Exception as e:
        logger.error(f"Ошибка создания модели в Trial {trial.number}: {e}")
        raise optuna.TrialPruned()

    num_params = model.get_num_params()
    logger.info(f"Trial {trial.number} - Num Parameters: {num_params:,}")
    trial.set_user_attr('num_params', num_params)
    
    # Создание DataLoader
    train_loader, val_loader = get_loaders(cfg)

    # Создаем папку для чекпоинтов этого trial
    trial_checkpoint_dir = os.path.join(cfg.train.save_dir, f"trial_{trial.number}")
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    trial.set_user_attr("checkpoint_dir", trial_checkpoint_dir)

    # Запуск тренировки
    checkpoint_dir = os.path.join(cfg.train.save_dir, f"trial_{trial.number}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        best_val_accuracy, avg_time_per_epoch = train_loop(model, train_loader, val_loader, cfg, checkpoint_dir, trial)
    except RuntimeError as e:
        logger.error(f"Ошибка тренировки в Trial {trial.number}: {e}")
        raise optuna.TrialPruned()

    trial.set_user_attr('avg_time_per_epoch', avg_time_per_epoch)

    # Сохраняем конфигурацию trial
    trial.set_user_attr('config', arch_params)

    # Вычисляем кастомный score
    score = compute_custom_score(best_val_accuracy, num_params, avg_time_per_epoch, cfg)
    logger.info(f"Trial {trial.number} (architecture) - Custom score: {score:.4f}, Accuracy: {best_val_accuracy:.4f}, Params: {num_params:,}, Time per epoch: {avg_time_per_epoch:.2f}s")

    return score

def objective_hyperparameter_search(trial, cfg, architecture_params):
    """
    Objective функция для подбора гиперпараметров.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Начало Trial {trial.number} для поиска гиперпараметров.")

    # Определяем тип модели
    model_type = cfg.model.custom_model_type if cfg.model.model_type == 'custom' else cfg.model.model_type

    # Устанавливаем архитектуру
    for k, v in architecture_params.items():
        setattr(cfg.model, k, v)

    # Валидация конфига
    if not hasattr(cfg.nas.hyperparameters_search, 'hyper_params'):
        raise ValueError("Конфиг NAS должен содержать 'nas.hyperparameters_search.hyper_params'.")

    # Генерируем гиперпараметры
    hyp_params = suggest_params(trial, cfg.nas.hyperparameters_search.hyper_params, prefix="hyp_")

    # Обновляем конфиг
    cfg.train.lr = hyp_params.get('lr', 1e-4)
    cfg.model.dropout = hyp_params.get('dropout', 0.1)
    cfg.train.weight_decay = hyp_params.get('weight_decay', 0.01)
    cfg.train.batch_size = hyp_params.get('batch_size', 8)
    cfg.train.epochs = cfg.nas.hyperparameters_search.epochs_per_trial

    print(f"\n   Конфигурация для Trial {trial.number} (гиперпараметры, {model_type}):")
    for k, v in hyp_params.items():
        print(f"      {k.capitalize()}: {v}")

    # Инициализация модели
    try:
        model = get_model(cfg)
        model.to(cfg.hardware.device)
    except Exception as e:
        logger.error(f"Ошибка создания модели в Trial {trial.number}: {e}")
        raise optuna.TrialPruned()

    num_params = model.get_num_params()
    logger.info(f"Trial {trial.number} - Num Parameters: {num_params:,}")
    trial.set_user_attr('num_params', num_params)
    
    # Создание DataLoader
    train_loader, val_loader = get_loaders(cfg)

    # Создаем папку для чекпоинтов этого trial
    trial_checkpoint_dir = os.path.join(cfg.train.save_dir, f"trial_{trial.number}")
    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    trial.set_user_attr("checkpoint_dir", trial_checkpoint_dir)

    # Запуск тренировки
    checkpoint_dir = os.path.join(cfg.train.save_dir, f"trial_{trial.number}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        best_val_accuracy, avg_time_per_epoch = train_loop(model, train_loader, val_loader, cfg, checkpoint_dir, trial)
    except RuntimeError as e:
        logger.error(f"Ошибка тренировки в Trial {trial.number}: {e}")
        raise optuna.TrialPruned()

    trial.set_user_attr('avg_time_per_epoch', avg_time_per_epoch)

    # Сохраняем конфигурацию trial
    trial_config = architecture_params.copy()
    trial_config.update(hyp_params)
    trial.set_user_attr('config', trial_config)

    # Вычисляем кастомный score
    score = compute_custom_score(best_val_accuracy, num_params, avg_time_per_epoch, cfg)
    logger.info(f"Trial {trial.number} (hyperparameters) - Custom score: {score:.4f}, Accuracy: {best_val_accuracy:.4f}, Params: {num_params:,}, Time per epoch: {avg_time_per_epoch:.2f}s")

    return score

def select_best_model(study, cfg):
    """
    Выбирает лучшую модель по кастомному score.
    """
    logger = logging.getLogger(__name__)
    best_trial = None
    best_score = -float('inf')

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        score = trial.value  # Value уже содержит кастомный score из objective
        if score > best_score:
            best_score = score
            best_trial = trial

    if best_trial is None:
        logger.warning("Нет завершенных trials. Возвращаем None.")
        return None

    logger.info(f"Выбрана лучшая модель: score={best_score:.4f}, config={best_trial.user_attrs['config']}")
    return best_trial

def main(config_path):
    """Основная функция для запуска NAS."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфиг файл не найден: {config_path}")

    # Загрузка конфигов из файлов для сравнения и определения имени
    base_cfg = OmegaConf.load("configs/base.yaml")
    model_cfg = OmegaConf.load(config_path)
    cfg_from_files = OmegaConf.merge(base_cfg, model_cfg)
    model_name_for_log = "unknown_model"
    if cfg_from_files.model.model_type == 'timm':
        model_name_for_log = cfg_from_files.model.timm_model_name
    elif cfg_from_files.model.model_type == 'custom':
        model_name_for_log = f"custom_{cfg_from_files.model.custom_model_type}"

    # Настройка логирования
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"log_{timestamp}_nas_{model_name_for_log}.log"
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    # --- Логика для создания/возобновления study ---
    optuna_storage_dir = "optuna_studies"
    os.makedirs(optuna_storage_dir, exist_ok=True)

    # Проверяем, есть ли информация о study в переданном конфиге
    if hasattr(cfg_from_files.nas, 'study_name') and hasattr(cfg_from_files.nas, 'study_db_path'):
        # --- ПУТЬ ВОЗОБНОВЛЕНИЯ ---
        study_name = cfg_from_files.nas.study_name
        study_db_file = cfg_from_files.nas.study_db_path
        cfg = cfg_from_files
        logger.info(f"Обнаружена информация в конфиге. Возобновление study '{study_name}' из файла {study_db_file}")
    else:
        # --- ПУТЬ СОЗДАНИЯ НОВОГО STUDY ---
        study_name = f"{os.path.basename(config_path).replace('.yaml', '')}_{model_name_for_log}"
        study_config_path = os.path.join(optuna_storage_dir, f'{study_name}_config.yaml')

        if os.path.exists(study_config_path):
            raise FileExistsError(f"Study с именем '{study_name}' уже существует. "
                                  f"Чтобы возобновить его, используйте сохраненный конфиг: "
                                  f"python nas.py --config {study_config_path}")
        
        logger.info(f"Создание нового study '{study_name}'.")
        cfg = cfg_from_files
        
        # Генерируем путь к БД и добавляем его и имя study в конфиг
        study_db_path = os.path.join(optuna_storage_dir, f'{study_name}.db')
        study_db_file = f"sqlite:///{study_db_path}"
        cfg.nas.study_name = study_name
        cfg.nas.study_db_path = study_db_file

        # Сохраняем дополненный конфиг для будущих возобновлений
        logger.info(f"Сохранение конфигурации для возобновления в {study_config_path}")
        OmegaConf.save(cfg, study_config_path)

    # Фиксация seed
    if cfg.train.manual_seed_enabled:
        SEED = cfg.train.seed
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    
    # Создаём основную папку для текущего эксперимента (с timestamp)
    experiment_name = f"{timestamp}_nas_{model_name_for_log}"
    experiment_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    cfg.train.save_dir = experiment_dir

    # Сохранение конфигурации для архива этого запуска
    cfg.nas.study_db_path = study_db_file
    config_path_saved = os.path.join(experiment_dir, "nas_config.yaml")
    with open(config_path_saved, 'w') as f:
        OmegaConf.save(cfg, f)
    
    logger.info("Starting NAS experiment with config saved at: %s", config_path_saved)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Optuna study results will be saved to: {study_db_file}")

    logger.info("\n" + "="*60 + "\n📚 ЭТАП 1: Поиск архитектуры\n" + "="*60)

    # Логируем фиксированные гиперпараметры для этого этапа
    try:
        logger.info("--- Фиксированные гиперпараметры для поиска архитектуры ---")
        logger.info(f"Learning Rate: {cfg.train.lr}")
        logger.info(f"Dropout: {cfg.model.dropout}")
        logger.info(f"Batch Size: {cfg.train.batch_size}")
        logger.info("----------------------------------------------------")
    except Exception as e:
        logger.error(f"Не удалось прочитать фиксированные гиперпараметры из конфига: {e}")
        raise

    # -- ЭТАП 1: Поиск архитектуры --

    # Получаем параметры для sampler из конфига
    arch_search_cfg = cfg.nas.architecture_search
    sampler_params_arch = arch_search_cfg.get('sampler_params', {})
    pruner_params_arch = arch_search_cfg.get('pruner_params', {})

    try:
        sampler_arch = getattr(optuna.samplers, arch_search_cfg.sampler)(**sampler_params_arch)
        pruner_arch = getattr(optuna.pruners, arch_search_cfg.pruner)(**pruner_params_arch)
        logger.info(f"🎯 Используем Sampler: {arch_search_cfg.sampler}, Pruner: {arch_search_cfg.pruner}")
    except AttributeError as e:
        logger.error(f"Неверное имя Sampler или Pruner в конфиге: {e}")
        raise

    study_params_arch = {
        "storage": study_db_file,
        "study_name": f"{study_name}_architecture",
        "direction": arch_search_cfg.direction,
        "sampler": sampler_arch,
        "pruner": pruner_arch,
        "load_if_exists": True
    }
    
    study_arch = optuna.create_study(**study_params_arch)
    study_arch.set_user_attr("config", OmegaConf.to_container(cfg, resolve=True))
    study_arch.optimize(
        lambda trial: objective_architecture_search(trial, cfg),
        n_trials=arch_search_cfg.num_trials,
        callbacks=[save_best_config_callback]
    )

    try:
        best_trial_arch = study_arch.best_trial
        logger.info(f"Поиск архитектуры завершен. Лучший score: {best_trial_arch.value:.4f}")
        best_architecture_params = {k.replace('arch_', ''): v for k, v in best_trial_arch.params.items()}
        logger.info(f"Используем архитектуру из лучшего trial для поиска гиперпараметров: {best_architecture_params}")
    except ValueError:
        logger.error("Не найдено ни одного успешно завершенного trial в поиске архитектуры. Невозможно продолжить поиск гиперпараметров.")
        logger.error("Если вы хотели запустить только поиск гиперпараметров, убедитесь, что в базе данных уже есть результаты поиска архитектуры.")
        return  # Завершаем выполнение, если не из чего выбрать

    logger.info("\n" + "="*60 + "\nЭТАП 2: Поиск гиперпараметров\n" + "="*60)

    # -- ЭТАП 2: Поиск гиперпараметров --

    # Получаем параметры для второго этапа
    hp_search_cfg = cfg.nas.hyperparameters_search
    sampler_params_hp = hp_search_cfg.get('sampler_params', {})
    pruner_params_hp = hp_search_cfg.get('pruner_params', {})
    
    try:
        sampler_hp = getattr(optuna.samplers, hp_search_cfg.sampler)(**sampler_params_hp)
        pruner_hp = getattr(optuna.pruners, hp_search_cfg.pruner)(**pruner_params_hp)
        logger.info(f"Используем Sampler: {hp_search_cfg.sampler}, Pruner: {hp_search_cfg.pruner}")
    except AttributeError as e:
        logger.error(f"Неверное имя Sampler или Pruner в конфиге: {e}")
        raise
        
    study_params_hp = {
        "storage": study_db_file,
        "study_name": f"{study_name}_hyperparameters",
        "direction": hp_search_cfg.direction,
        "sampler": sampler_hp,
        "pruner": pruner_hp,
        "load_if_exists": True
    }

    study_hp = optuna.create_study(**study_params_hp)
    study_hp.set_user_attr("config", OmegaConf.to_container(cfg, resolve=True))
    study_hp.optimize(
        lambda trial: objective_hyperparameter_search(trial, cfg, best_architecture_params),
        n_trials=hp_search_cfg.num_trials,
        callbacks=[save_best_config_callback]
    )

    best_hyp_trial = select_best_model(study_hp, cfg)
    if best_hyp_trial:
        best_hyperparameters = best_hyp_trial.user_attrs.get('config', {})
        logger.info(f"Лучшие гиперпараметры: {best_hyperparameters} с score {best_hyp_trial.value:.4f}")
        print("\nАнализ результатов поиска гиперпараметров:")
        print(f"Лучшие гиперпараметры: {best_hyperparameters}")
        print(f"   Score: {best_hyp_trial.value:.4f}")

    logger.info("NAS experiment completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neural Architecture Search for Food101.")
    parser.add_argument("--config", type=str, default="configs/nas_vit.yaml",
                        help="Path to the NAS configuration file.")
    args = parser.parse_args()
    main(args.config)
    
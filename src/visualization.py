import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from omegaconf import OmegaConf
import logging

def plot_accuracy_curves(checkpoint_dir, epochs, train_accuracies, val_accuracies, cfg, logger):
    """Plot training and validation accuracy curves over epochs."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train accuracy')
    plt.plot(epochs, val_accuracies, label='Val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'accuracy_curves.png'))
    plt.close()
    logger.info(f"Saved accuracy curves to {os.path.join(graphs_dir, 'accuracy_curves.png')}")

def plot_aggregated_metrics(checkpoint_dir, epochs, cfg, logger):
    """Plot macro and micro F1-score over epochs."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    macro_f1, micro_f1 = [], []
    num_classes = cfg.dataset.num_classes

    for epoch in range(len(epochs)):
        val_labels_path = os.path.join(checkpoint_dir, 'labels', f'epoch_{epoch+1}_val.npy')
        if not os.path.exists(val_labels_path):
            logger.warning(f"Validation labels not found for epoch {epoch+1} at {val_labels_path}")
            continue
        val_data = np.load(val_labels_path)
        true_labels = val_data[:, 0]
        pred_labels = val_data[:, 1]
        _, _, f1_macro, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0.0)
        _, _, f1_micro, _ = precision_recall_fscore_support(true_labels, pred_labels, average='micro', zero_division=0.0)
        macro_f1.append(f1_macro)
        micro_f1.append(f1_micro)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, macro_f1, label='Macro F1')
    plt.plot(epochs, micro_f1, label='Micro F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Aggregated Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'aggregated_metrics.png'))
    plt.close()
    logger.info(f"Saved aggregated metrics to {os.path.join(graphs_dir, 'aggregated_metrics.png')}")

def plot_worst_classes_confusion(checkpoint_dir, last_epoch, cfg, logger):
    """Plot confusion heatmap for the 20 worst classes based on F1-score."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    val_labels_path = os.path.join(checkpoint_dir, 'labels', f'epoch_{last_epoch}_val.npy')
    if not os.path.exists(val_labels_path):
        logger.warning(f"Validation labels not found for epoch {last_epoch} at {val_labels_path}")
        return
    val_data = np.load(val_labels_path)
    true_labels = val_data[:, 0]
    pred_labels = val_data[:, 1]
    num_classes = cfg.dataset.num_classes
    
    # Загружаем имена классов из локального файла
    try:
        if not hasattr(cfg.dataset, 'path'):
            raise ValueError("В конфиге (configs/base.yaml) не указан путь к датасету: dataset.path")
        classes_file_path = os.path.join(cfg.dataset.path, 'food-101', 'meta', 'classes.txt')
        with open(classes_file_path, 'r') as f:
            class_names = [line.strip().replace('_', ' ') for line in f.readlines()]
    except Exception as e:
        logger.error(f"Не удалось загрузить имена классов из {classes_file_path}: {e}")
        # Создаем заглушки, если файл не найден, чтобы визуализация не падала
        class_names = [f'class_{i}' for i in range(num_classes)]


    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) # Добавлено 1e-6 для стабильности
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(num_classes), zero_division=0)

    worst_indices = np.argsort(f1)[:20]
    worst_labels = [class_names[i] for i in worst_indices]

    fig, axs = plt.subplots(5, 4, figsize=(20, 20))
    fig.suptitle(f'Top 20 Worst Classes Confusion (Epoch {last_epoch})', y=1.01)

    for i, class_id in enumerate(worst_indices):
        row = i // 4
        col = i % 4
        confusions = cm_norm[class_id]
        confusions[class_id] = 0
        top_confusions = np.argsort(confusions)[-10:][::-1]
        top_values = confusions[top_confusions]
        top_names = [class_names[j] for j in top_confusions]

        x = np.arange(len(top_names))
        width = 0.25

        axs[row, col].bar(x, top_values, width, label='Confusion Probability', color='blue')
        axs[row, col].set_title(f'{worst_labels[i]} (Class {class_id})')
        axs[row, col].set_xlabel('Top Confused Classes')
        axs[row, col].set_ylabel('Normalized Confusion Value')
        axs[row, col].set_xticks(x)
        axs[row, col].set_xticklabels(top_names, rotation=90)
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'worst_classes_confusion_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved worst classes confusion to {os.path.join(graphs_dir, f'worst_classes_confusion_epoch_{last_epoch}.png')}")

def plot_all_classes_metrics(checkpoint_dir, last_epoch, cfg, logger):
    """Plot metrics for all classes sorted by Recall, Precision, and F1-score."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    val_labels_path = os.path.join(checkpoint_dir, 'labels', f'epoch_{last_epoch}_val.npy')
    if not os.path.exists(val_labels_path):
        logger.warning(f"Validation labels not found for epoch {last_epoch} at {val_labels_path}")
        return
    val_data = np.load(val_labels_path)
    true_labels = val_data[:, 0]
    pred_labels = val_data[:, 1]
    num_classes = cfg.dataset.num_classes

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(num_classes), zero_division=0)

    # Sorted by Recall
    sorted_indices = np.argsort(recall)
    sorted_precision_recall = precision[sorted_indices]
    sorted_recall_recall = recall[sorted_indices]
    sorted_f1_recall = f1[sorted_indices]
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_classes), sorted_recall_recall, label='Recall')
    plt.plot(range(num_classes), sorted_precision_recall, label='Precision')
    plt.plot(range(num_classes), sorted_f1_recall, label='F1-score')
    plt.xlabel('Classes (sorted by Recall)')
    plt.ylabel('Metric Value')
    plt.title('All Classes Metrics (Sorted by Recall)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f'all_classes_metrics_recall_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved all classes metrics (sorted by Recall) to {os.path.join(graphs_dir, f'all_classes_metrics_recall_epoch_{last_epoch}.png')}")

    # Sorted by Precision
    sorted_indices = np.argsort(precision)
    sorted_precision_precision = precision[sorted_indices]
    sorted_recall_precision = recall[sorted_indices]
    sorted_f1_precision = f1[sorted_indices]
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_classes), sorted_recall_precision, label='Recall')
    plt.plot(range(num_classes), sorted_precision_precision, label='Precision')
    plt.plot(range(num_classes), sorted_f1_precision, label='F1-score')
    plt.xlabel('Classes (sorted by Precision)')
    plt.ylabel('Metric Value')
    plt.title('All Classes Metrics (Sorted by Precision)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f'all_classes_metrics_precision_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved all classes metrics (sorted by Precision) to {os.path.join(graphs_dir, f'all_classes_metrics_precision_epoch_{last_epoch}.png')}")

    # Sorted by F1-score
    sorted_indices = np.argsort(f1)
    sorted_precision_f1 = precision[sorted_indices]
    sorted_recall_f1 = recall[sorted_indices]
    sorted_f1_f1 = f1[sorted_indices]
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_classes), sorted_recall_f1, label='Recall')
    plt.plot(range(num_classes), sorted_precision_f1, label='Precision')
    plt.plot(range(num_classes), sorted_f1_f1, label='F1-score')
    plt.xlabel('Classes (sorted by F1-score)')
    plt.ylabel('Metric Value')
    plt.title('All Classes Metrics (Sorted by F1-score)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f'all_classes_metrics_f1_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved all classes metrics (sorted by F1-score) to {os.path.join(graphs_dir, f'all_classes_metrics_f1_epoch_{last_epoch}.png')}")

def plot_boxplot_metrics(checkpoint_dir, last_epoch, cfg, logger):
    """Plot boxplot distribution of precision, recall, and F1-score."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    val_labels_path = os.path.join(checkpoint_dir, 'labels', f'epoch_{last_epoch}_val.npy')
    if not os.path.exists(val_labels_path):
        logger.warning(f"Validation labels not found for epoch {last_epoch} at {val_labels_path}")
        return
    val_data = np.load(val_labels_path)
    true_labels = val_data[:, 0]
    pred_labels = val_data[:, 1]
    num_classes = cfg.dataset.num_classes

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(num_classes), zero_division=0)

    plt.figure(figsize=(10, 6))
    plt.boxplot([precision, recall, f1], tick_labels=['Precision', 'Recall', 'F1-score'])
    plt.title('Distribution of Metrics Across All Classes')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f'boxplot_metrics_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved boxplot metrics to {os.path.join(graphs_dir, f'boxplot_metrics_epoch_{last_epoch}.png')}")

def plot_cumulative_f1(checkpoint_dir, last_epoch, cfg, logger):
    """Plot cumulative gain curve of F1-score."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    val_labels_path = os.path.join(checkpoint_dir, 'labels', f'epoch_{last_epoch}_val.npy')
    if not os.path.exists(val_labels_path):
        logger.warning(f"Validation labels not found for epoch {last_epoch} at {val_labels_path}")
        return
    val_data = np.load(val_labels_path)
    true_labels = val_data[:, 0]
    pred_labels = val_data[:, 1]
    num_classes = cfg.dataset.num_classes

    _, _, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(num_classes), zero_division=0)
    sorted_f1_indices = np.argsort(f1)[::-1]
    cumulative_f1 = np.cumsum(np.sort(f1)[::-1]) / np.sum(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_classes), cumulative_f1, label='Cumulative F1-score')
    plt.xlabel('Number of Classes (sorted by F1-score)')
    plt.ylabel('Cumulative F1-score Proportion')
    plt.title('Cumulative Gain Curve of F1-score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f'cumulative_f1_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved cumulative F1-score to {os.path.join(graphs_dir, f'cumulative_f1_epoch_{last_epoch}.png')}")

def plot_scatter_metrics(checkpoint_dir, last_epoch, cfg, logger):
    """Plot scatter plot of Precision vs Recall with F1-score as color."""
    graphs_dir = os.path.join(checkpoint_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    val_labels_path = os.path.join(checkpoint_dir, 'labels', f'epoch_{last_epoch}_val.npy')
    if not os.path.exists(val_labels_path):
        logger.warning(f"Validation labels not found for epoch {last_epoch} at {val_labels_path}")
        return
    val_data = np.load(val_labels_path)
    true_labels = val_data[:, 0]
    pred_labels = val_data[:, 1]
    num_classes = cfg.dataset.num_classes

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(num_classes), zero_division=0)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(precision, recall, c=f1, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, label='F1-score')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall for All Classes')
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f'scatter_metrics_epoch_{last_epoch}.png'))
    plt.close()
    logger.info(f"Saved scatter metrics to {os.path.join(graphs_dir, f'scatter_metrics_epoch_{last_epoch}.png')}")

def generate_plots(checkpoint_dir, cfg, logger):
    """Generate all visualization plots after training."""
    logger.info(f"Starting visualization for checkpoint directory: {checkpoint_dir}")

    metrics_file = os.path.join(checkpoint_dir, 'metrics.txt')
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file not found at {metrics_file}")
        return

    # Чтение метрик
    epochs = []
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    with open(metrics_file, 'r') as f:
        next(f)  # Skip header
        for line in f:
            epoch, train_loss, train_acc, val_loss, val_acc = map(float, line.strip().split(','))
            epochs.append(int(epoch) - 1)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

    # Последняя эпоха
    last_epoch = epochs[-1] + 1 if epochs else 1

    # Генерация всех графиков
    plot_accuracy_curves(checkpoint_dir, epochs, train_accs, val_accs, cfg, logger)
    plot_aggregated_metrics(checkpoint_dir, epochs, cfg, logger)
    plot_worst_classes_confusion(checkpoint_dir, last_epoch, cfg, logger)
    plot_all_classes_metrics(checkpoint_dir, last_epoch, cfg, logger)
    plot_boxplot_metrics(checkpoint_dir, last_epoch, cfg, logger)
    plot_cumulative_f1(checkpoint_dir, last_epoch, cfg, logger)
    plot_scatter_metrics(checkpoint_dir, last_epoch, cfg, logger)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils.config import CLASSES

def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss por Época', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_title('Accuracy por Época', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_per_class_metrics(per_class_metrics, save_path=None):
    classes = list(per_class_metrics.keys())
    metrics_names = ['precision', 'recall', 'f1_score']
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics_names):
        values = [per_class_metrics[c][metric] for c in classes]
        plt.bar(x + i * width, values, width, label=metric.capitalize())

    plt.xlabel('Clase')
    plt.ylabel('Valor')
    plt.title('Métricas por Clase', fontsize=14, fontweight='bold')
    plt.xticks(x + width, classes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

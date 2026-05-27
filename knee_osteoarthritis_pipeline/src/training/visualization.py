import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names=None, save_path="confusion_matrix.png"):
    """
    Vẽ Error Heatmap cho Confusion Matrix để xem mô hình có bị nhầm lẫn giữa
    các mức độ bệnh hay không (ví dụ: Level 2 bị nhầm thành Level 3).
    """
    plt.figure(figsize=(10, 8))
    if class_names is None:
        class_names = [f"Level {i}" for i in range(cm.shape[0])]
        
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Best Epoch)')
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán (Predicted Label)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_history(history, save_path="training_history.png"):
    """
    Vẽ biểu đồ lịch sử Loss và F1-Macro qua từng Epoch.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 6))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Biểu đồ Macro F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-o', label='Train F1-Macro')
    plt.plot(epochs, history['val_f1'], 'r-o', label='Validation F1-Macro')
    plt.title('Training & Validation F1-Macro')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
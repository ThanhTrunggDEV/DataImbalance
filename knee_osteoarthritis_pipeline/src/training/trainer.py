import torch
import numpy as np
import os
from tqdm import tqdm
from .metrics import calculate_metrics
from .visualization import plot_confusion_matrix, plot_training_history

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, lr_scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        loop = tqdm(self.train_loader, desc="Training")
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients (common in large efficientnets)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
            
        metrics = calculate_metrics(all_labels, all_preds)
        return total_loss / len(self.train_loader), metrics
        
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validation")
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs))
        return total_loss / len(self.val_loader), metrics
        
    def fit(self, epochs, save_dir="../checkpoints"):
        best_f1 = 0.0
        best_cm = None
        
        # Lưu trữ lịch sử để vẽ biểu đồ
        history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, "efficientnet_b5_best.pth")
        
        for epoch in range(epochs):
            print(f"\n[{epoch+1}/{epochs}]")
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            # Ghi nhận lịch sử
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_f1'].append(train_metrics['f1_macro'])
            history['val_f1'].append(val_metrics['f1_macro'])
            
            if self.lr_scheduler:
                # Step scheduler based on validation loss
                self.lr_scheduler.step(val_loss) 
                
            print(f"Train Loss: {train_loss:.4f} | Train Macro-F1: {train_metrics['f1_macro']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_metrics['f1_macro']:.4f} | Val Macro-AUC: {val_metrics.get('auc_macro', 0):.4f}")
            
            # Medical scenarios with imbalanced data prioritize Macro F1 
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                best_cm = val_metrics['confusion_matrix']
                torch.save(self.model.state_dict(), model_save_path)
                print(f"--> Saved best model with Macro-F1: {best_f1:.4f}")
                
        print("\nTraining completed! Generating evaluation plots...")
        # Kết thúc training, vẽ biểu đồ
        plot_training_history(history, save_path=os.path.join(save_dir, "training_history.png"))
        if best_cm is not None:
            plot_confusion_matrix(best_cm, save_path=os.path.join(save_dir, "best_confusion_matrix.png"))
        print(f"Plots saved to {save_dir}/")
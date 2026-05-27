import os
import torch
import torch.optim as optim
import sys

# Thêm đường dẫn để python có thể tìm thấy các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.download import download_kaggle_dataset
from data.dataset import get_dataloaders
from models.efficientnet import get_efficientnet_model
from training.loss import FocalLoss
from training.trainer import Trainer

def main():
    # ==========================
    # Cấu hình tham số
    # ==========================
    dataset_name = "shashwatwork/knee-osteoarthritis-dataset-with-severity"
    data_dir = "../data" # Tùy chỉnh theo cách giải nén (có thể cần sửa path nếu kaggle tự tạo folder con)
    
    # EfficientNet-B5 khá nặng, sử dụng batch size nhỏ (phụ thuộc vào VRAM GPU)
    batch_size = 8 
    epochs = 30
    lr = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Tải Dataset
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        download_kaggle_dataset(dataset_name, data_dir)
        
    # *Lưu ý*: Folder của bộ dữ liệu cấu trúc thực tế sau khi unzip có thể nằm trong thư mục con như: `../data/Data/train`
    # Hãy điều chỉnh biến `actual_data_root` nêú cần.
    actual_data_root = os.path.join(data_dir, "Data") if os.path.exists(os.path.join(data_dir, "Data")) else data_dir
        
    # 2. Chuẩn bị DataLoader (Áp dụng WeightedRandomSampler)
    print("Preparing Dataloaders with Image size 456x456 (EfficientNet-B5)...")
    train_loader, val_loader, test_loader, raw_class_weights = get_dataloaders(actual_data_root, batch_size=batch_size)
    
    # 3. Chuyển đổi trọng số lớp thành Tensor để dùng cho Focal Loss
    # Chuẩn hóa weights để tổng bằng 1 (hoặc theo công thức của bạn)
    class_weights_tensor = torch.tensor(raw_class_weights, dtype=torch.float32).to(device)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()
    print(f"Class Weights calculated: {class_weights_tensor.cpu().numpy()}")

    # 4. Khởi tạo Model
    print("Initializing EfficientNet-B5 Model...")
    model = get_efficientnet_model(num_classes=5, pretrained=True).to(device)
    
    # 5. Khởi tạo Hàm Loss và Optimizer
    # Sử dụng Class-Weighted Focal Loss để giải quyết triệt để mất cân bằng
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    
    # AdamW là optimizer hoạt động rất tốt đối với các mô hình vision hạng nặng kết hợp weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning Rate Scheduler - Tự động giảm LR nếu validation loss chững lại (Plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    # 6. Pipeline Huấn luyện
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler
    )
    
    if not os.path.exists("../checkpoints"):
        os.makedirs("../checkpoints")
        
    trainer.fit(epochs=epochs, save_dir="../checkpoints")

if __name__ == "__main__":
    main()
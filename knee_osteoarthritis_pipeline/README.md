# Pipeline Huấn luyện EfficientNet-B5 cho Y tế với Dữ liệu Mất cân bằng (Knee Osteoarthritis)

Mã nguồn này được module hóa để giải quyết bài toán phân loại mức độ thoái hóa khớp gối từ hình ảnh X-Quang y tế với tình trạng **mất cân bằng class cực kỳ nghiêm trọng** (vd: lớp bệnh nặng rất ít so với lớp bình thường).

## Các Kỹ thuật xử lý mất cân bằng tích hợp:
1. **WeightedRandomSampler (`dataset.py`)**: Đảm bảo mỗi batch sinh ra có tỷ lệ ảnh của các class thiểu số tương đương với các class đa số, giúp mô hình "thấy" đều các độ nghiêm trọng của bệnh.
2. **Weighted Focal Loss (`loss.py`)**: Hàm mất mát Focal Loss tập trung bắt mô hình học các ca khó (loss bự), kết hợp thiết lập `alpha` (Class Weights) phạt mạnh khi đoán sai trên các nhóm bệnh thiểu số.
3. **Macro-Average Metrics (`metrics.py`)**: Sử dụng F1-Macro và AUC-Macro, không sử dụng Accuracy đơn thuần (để tránh thiên kiến với class chiếm đa số).
4. **Data Augmentation (`dataset.py`)**: Lật ảnh, xoay góc nhẹ, thay đổi độ sáng để tạo đa dạng từ tập mẫu sinh ngẫu nhiên.

## Hướng dẫn cài đặt
```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate

# Cài đặt requirements
pip install -r requirements.txt
```

*(Lưu ý: Để module tự tải dữ liệu Kaggle ở file download.py, bạn cần có file `kaggle.json` hợp lệ trong `%USERPROFILE%\.kaggle\)*

## Chạy Pipeline
```bash
cd src
python main.py
```
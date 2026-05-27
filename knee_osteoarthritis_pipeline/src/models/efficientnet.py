import timm

def get_efficientnet_model(num_classes=5, pretrained=True):
    """
    Load EfficientNet-B5 with NoisyStudent weights (often better for complex images).
    Sử dụng API chuẩn của timm (tham số num_classes và drop_rate) để tự động
    điều chỉnh head cho bài toán 5 lớp và bật Dropout chống Overfit.
    """
    # tf_efficientnet_b5_ns uses NoisyStudent pre-trained weights
    model = timm.create_model(
        'tf_efficientnet_b5_ns', 
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.5 # Adding heavy Dropout to combat overfitting
    )
    
    return model
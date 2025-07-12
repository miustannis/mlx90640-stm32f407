import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split

class TemperatureDataProcessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.min_temp = 10  
        self.max_temp = 40  

    def load_raw_data(self, file_path):
        """加载原始数据并解析为numpy数组
        改进点：
        1. 处理空值/非法字符
        2. 自动跳过非数值内容
        3. 更清晰的错误提示
        4. 支持不完整数据
        """
        with open(file_path, 'r') as f:
            raw_data = f.read().strip()
        
        
        values = []
        error_count = 0
        for x in raw_data.replace('\n', ',').split(','):  
            x = x.strip()
            if not x:  
                continue
            try:
                values.append(float(x))
            except ValueError:
                error_count += 1
        
        if error_count > 0:
            print(f"警告: 发现并跳过 {error_count} 个非法数据点")
        
        all_values = np.array(values)
        expected_points_per_frame = 32 * 24
        total_frames = len(all_values) // expected_points_per_frame
        
        
        if len(all_values) < expected_points_per_frame:
            raise ValueError(
                f"数据不足！至少需要 {expected_points_per_frame} 个点，"
                f"但只得到 {len(all_values)} 个"
            )
        
        if len(all_values) % expected_points_per_frame != 0:
            truncated = len(all_values) % expected_points_per_frame
            print(
                f"警告: 最后 {truncated} 个点不构成完整帧，"
                f"已自动截断 (总点数: {len(all_values)})"
            )
        
        # 重塑为(帧数, 高度, 宽度)
        valid_points = total_frames * expected_points_per_frame
        frames = all_values[:valid_points].reshape(-1, 32, 24)
        
        print(
            f"成功加载 {total_frames} 帧数据 "
            f"(有效点数: {valid_points}/{len(all_values)})"
        )
        return frames

    def clean_data(self, data):
        """数据清洗"""
        cleaned = data.copy()
        
        # 去除异常值 (基于物理合理范围)
        cleaned = np.clip(cleaned, self.min_temp, self.max_temp)
        
        # 无效帧检测 (根据温度方差)
        frame_vars = np.var(cleaned, axis=(1,2))
        valid_mask = frame_vars > 1.0  # 方差阈值
        
        print(f"移除{len(data) - sum(valid_mask)}无效帧 (低方差)")
        return cleaned[valid_mask]

    def normalize(self, data):
        """基于统计量标准化"""
        if self.mean is None or self.std is None:
            self.mean = np.mean(data)
            self.std = np.std(data)
            print(f"计算归一化参数: mean={self.mean:.2f}, std={self.std:.2f}")
        
        return (data - self.mean) / (self.std + 1e-8)

    def create_dataset(self, all_frames, labels, augment=True):
        """创建带增强的数据集"""
        # 转换为torch张量
        frames_tensor = torch.FloatTensor(all_frames)
        labels_tensor = torch.LongTensor(labels)
        
        # 数据增强 (仅在训练时)
        if augment:
            augmented = []
            for frame in frames_tensor:
                augmented.append(self.augment_frame(frame))
                # 添加原始数据
                augmented.append(frame)  
            
            frames_tensor = torch.stack(augmented)
            # 对应标签也要扩展
            labels_tensor = labels_tensor.repeat_interleave(2)
        
        return frames_tensor, labels_tensor

    def augment_frame(self, frame):
        """温度数据专用增强"""
        # 添加噪声 (保持温度物理意义)
        noise = torch.randn(32, 24) * 0.3  # 标准差0.3℃
        noisy_frame = frame + noise
        
        # 模拟热扩散 (高斯模糊)
        from scipy.ndimage import gaussian_filter
        blurred = torch.FloatTensor(
            gaussian_filter(noisy_frame.numpy(), sigma=0.8))
        
        # 随机遮挡 (模拟部分遮挡)
        if torch.rand(1) > 0.7:
            h, w = torch.randint(3,8,(2,))
            y, x = torch.randint(0,32-h,(1,)), torch.randint(0,24-w,(1,))
            blurred[y:y+h, x:x+w] = torch.mean(blurred)
        
        return blurred

# 使用示例
if __name__ == "__main__":
    # 假设有5个文件，每个文件对应一类手势
    data_files = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]
    
    processor = TemperatureDataProcessor()
    all_frames = []
    labels = []
    
    # 加载并标记所有数据
    for label, file_path in enumerate(data_files):
        frames = processor.load_raw_data(file_path)
        cleaned = processor.clean_data(frames)
        all_frames.append(cleaned)
        labels.extend([label] * len(cleaned))
        print(f"加载类别{label}: {len(cleaned)}有效帧")
    
    # 合并所有数据
    X = np.concatenate(all_frames)
    y = np.array(labels)
    
    # 标准化处理
    X_normalized = processor.normalize(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y)
    
    # 创建增强后的数据集
    X_train_tensor, y_train_tensor = processor.create_dataset(X_train, y_train)
    X_test_tensor, y_test_tensor = processor.create_dataset(X_test, y_test, augment=False)
    
    print("\n最终数据集:")
    print(f"训练集: {X_train_tensor.shape} (增强后), 标签: {y_train_tensor.shape}")
    print(f"测试集: {X_test_tensor.shape}, 标签: {y_test_tensor.shape}")
    
    # 保存预处理参数 (部署时需要)
    np.savez("preprocess_params.npz", 
             mean=processor.mean, 
             std=processor.std,
             min_temp=processor.min_temp,
             max_temp=processor.max_temp)
    
        # 保存预处理数据集
    os.makedirs("processed_data", exist_ok=True)

    # 保存为PyTorch格式
    torch.save({
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor
    }, "processed_data/temp_dataset.pt")

    # 保存为NumPy格式
    np.savez("processed_data/temp_dataset.npz",
            X_train=X_train_tensor.numpy(),
            y_train=y_train_tensor.numpy(),
            X_test=X_test_tensor.numpy(),
            y_test=y_test_tensor.numpy())

    print("\n预处理数据已保存至 processed_data/ 目录：")
    print("├── temp_dataset.pt      # PyTorch格式")
    print("├── temp_dataset.npz     # NumPy格式")
    print("└── preprocess_params.npz  # 标准化参数")
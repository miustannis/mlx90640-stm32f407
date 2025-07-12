import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 加载预处理好的数据
def load_processed_data(data_path="processed_data/temp_dataset.pt"):
    data = torch.load(data_path)
    return (
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )



class MicroTempNet(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()
        # 特征提取
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # [4, 16, 12]
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # [8, 8, 6]
            nn.ReLU(),
        )
        
        # 分类器 (参数量: 8*8*6*8 + 8 + 8*5 + 5 = 2,669)
        self.classifier = nn.Sequential(
            nn.Linear(8*8*6, 8),  
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度 [B,1,32,24]
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 训练和部署工具函数
class TempNetUtils:
    @staticmethod
    def train(model, epochs=30):

                # 加载预处理数据
        X_train, y_train, X_test, y_test = load_processed_data()
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False
        )

        """ 精简训练流程 """
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                running_loss += loss.item()
            
            # 计算epoch平均loss和准确率
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            print(f'Epoch {epoch+1}/{epochs} | '
                f'Loss: {epoch_loss:.4f} | '
                f'Train Acc: {epoch_acc:.2f}%')
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f'Val Loss: {val_loss/len(test_loader):.4f} | '
            f'Val Acc: {val_acc:.2f}%')

    @staticmethod
    def convert_to_onnx(model, output_path="tempnet.onnx"):
        """ 导出为ONNX格式 """
        dummy_input = torch.randn(1, 32, 24)
        torch.onnx.export(model, dummy_input, output_path,
                         input_names=['temp_input'],
                         output_names=['gesture_output'],
                         opset_version=11)

def export_weights(model, filename):
    """导出PyTorch模型的权重到C头文件"""
    weights = {}
    
    # 导出各层权重
    layers = [
        ('conv1', model.features[0]),
        ('conv2', model.features[2]),
        ('fc1', model.classifier[0]),
        ('fc2', model.classifier[2])
    ]
    
    for name, layer in layers:
        weights[f'{name}_weight'] = layer.weight.detach().numpy().flatten()
        weights[f'{name}_bias'] = layer.bias.detach().numpy().flatten()
    
    # 写入头文件
    with open(filename, 'w') as f:
        f.write('#ifndef MICRO_TEMP_NET_WEIGHTS_H\n')
        f.write('#define MICRO_TEMP_NET_WEIGHTS_H\n\n')
        
        for name, array in weights.items():
            f.write(f'const float {name}[] = {{\n')
            # 每行8个数值
            for i in range(0, len(array), 8):
                line = ', '.join([f'{x:.8f}f' for x in array[i:i+8]])
                f.write(f'    {line},\n')
            f.write('};\n\n')
        
        f.write('#endif // MICRO_TEMP_NET_WEIGHTS_H\n')


if __name__ == "__main__":
    # 初始化网络
    model = MicroTempNet(num_classes=5)
    print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练网络
    TempNetUtils.train(model, epochs=20)
    # 只保存state_dict
    torch.save(model.state_dict(), 'micro_temp_net_params.pth')
 
    # 导出为部署格式
    TempNetUtils.convert_to_onnx(model)

    model.load_state_dict(torch.load('micro_temp_net_params.pth'))
    export_weights(model, 'micro_temp_net_weights.h')
    

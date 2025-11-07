import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18

def count_parameters(model):
    """
    计算模型的总参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model):
    """
    获取模型的详细参数信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        dict: 包含模型参数信息的字典
    """
    total_params = 0
    trainable_params = 0
    
    layer_info = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        layer_info.append({
            'layer_name': name,
            'shape': list(param.shape),
            'num_params': num_params,
            'trainable': param.requires_grad
        })
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'layer_details': layer_info
    }

def print_model_summary(model):
    """
    打印模型参数摘要
    
    Args:
        model: PyTorch模型
    """
    info = get_model_info(model)
    
    print("=" * 70)
    print("模型参数摘要")
    print("=" * 70)
    print(f"总参数量:          {info['total_params']:,}")
    print(f"可训练参数量:      {info['trainable_params']:,}")
    print(f"不可训练参数量:    {info['non_trainable_params']:,}")
    print("=" * 70)
    
    print("\n各层参数详情:")
    print("-" * 70)
    print(f"{'层名称':<40} {'形状':<20} {'参数量':<15}")
    print("-" * 70)
    
    for layer in info['layer_details']:
        shape_str = str(layer['shape'])
        print(f"{layer['layer_name']:<40} {shape_str:<20} {layer['num_params']:<15,}")
    
    print("=" * 70)

def get_model_params_for_architecture(num_channels, num_classes):
    """
    根据模型架构获取参数量(不需要实际训练)
    
    Args:
        num_channels (int): 输入通道数
        num_classes (int): 分类类别数
    
    Returns:
        dict: 模型参数信息
    """
    model = CustomMultiChannelResNet18(num_channels=num_channels, num_classes=num_classes)
    return get_model_info(model)

# 示例使用
if __name__ == "__main__":
    # 创建一个示例模型
    model = CustomMultiChannelResNet18(num_channels=2, num_classes=8)
    
    # 打印模型摘要
    print_model_summary(model)
    
    # 或者只获取参数数量
    param_count = count_parameters(model)
    print(f"\n模型总参数量: {param_count:,}")
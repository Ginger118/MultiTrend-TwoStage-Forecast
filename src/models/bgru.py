"""
双向门控循环单元(BGRU)模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BGRUModel(nn.Module):
    """双向门控循环单元模型"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        """
        初始化BGRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
        """
        super(BGRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 双向GRU层
        self.bgru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 因为是双向
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 预测结果，形状为 (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # BGRU前向传播
        out, hn = self.bgru(x, h0)
        
        # 取最后一个时间步的输出
        # out形状: (batch_size, sequence_length, hidden_size * 2)
        last_output = out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # 应用dropout
        last_output = self.dropout_layer(last_output)
        
        # 全连接层
        output = self.fc(last_output)  # (batch_size, 1)
        
        return output

class EnhancedBGRU(nn.Module):
    """增强版BGRU模型，包含残差连接和层归一化"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(EnhancedBGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # 多个BGRU层
        self.bgru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_size if i == 0 else hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            ) for i in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(num_layers)
        ])
        
        # Dropout层
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 预测结果，形状为 (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # 通过多个BGRU层
        for i, (bgru, layer_norm, dropout) in enumerate(
            zip(self.bgru_layers, self.layer_norms, self.dropout_layers)
        ):
            # 保存残差连接的输入
            residual = x
            
            # BGRU前向传播
            out, _ = bgru(x)
            
            # 层归一化
            out = layer_norm(out)
            
            # 残差连接（如果维度匹配）
            if i > 0 and residual.size(-1) == out.size(-1):
                out = out + residual
            
            # Dropout
            out = dropout(out)
            
            # 更新x用于下一层
            x = out
        
        # 取最后一个时间步的输出
        last_output = x[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # 输出投影
        output = self.output_projection(last_output)  # (batch_size, 1)
        
        return output

class BGRUWithAttention(nn.Module):
    """带注意力机制的BGRU模型"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(BGRUWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BGRU层
        self.bgru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 预测结果，形状为 (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # BGRU前向传播
        bgru_out, _ = self.bgru(x, h0)
        
        # 自注意力机制
        attn_out, _ = self.attention(bgru_out, bgru_out, bgru_out)
        
        # 残差连接和层归一化
        out = self.layer_norm(bgru_out + attn_out)
        
        # Dropout
        out = self.dropout(out)
        
        # 全局平均池化
        pooled = torch.mean(out, dim=1)  # (batch_size, hidden_size * 2)
        
        # 输出层
        output = self.fc(pooled)  # (batch_size, 1)
        
        return output

def test_bgru_models():
    """测试BGRU模型"""
    print("测试BGRU模型...")
    
    # 设置参数
    batch_size = 32
    sequence_length = 30
    input_size = 7  # 6个百度指数关键词 + 1个ILI病例数
    hidden_size = 64
    num_layers = 2
    
    # 创建测试数据
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # 测试基础BGRU模型
    print("\n测试基础BGRU模型...")
    model1 = BGRUModel(input_size, hidden_size, num_layers)
    output1 = model1(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    print(f"参数数量: {sum(p.numel() for p in model1.parameters())}")
    
    # 测试增强版BGRU模型
    print("\n测试增强版BGRU模型...")
    model2 = EnhancedBGRU(input_size, hidden_size, num_layers)
    output2 = model2(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output2.shape}")
    print(f"参数数量: {sum(p.numel() for p in model2.parameters())}")
    
    # 测试带注意力的BGRU模型
    print("\n测试带注意力的BGRU模型...")
    model3 = BGRUWithAttention(input_size, hidden_size, num_layers)
    output3 = model3(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output3.shape}")
    print(f"参数数量: {sum(p.numel() for p in model3.parameters())}")
    
    print("\nBGRU模型测试完成！")

# 测试代码已移至 tests/ 目录

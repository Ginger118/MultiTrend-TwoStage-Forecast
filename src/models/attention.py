"""
多重注意力机制实现
包括SE注意力、通道注意力、空间注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SEAttention(nn.Module):
    """Squeeze and Excitation注意力机制"""
    
    def __init__(self, channels, reduction=16):
        """
        初始化SE注意力
        
        Args:
            channels: 输入通道数
            reduction: 降维比例
        """
        super(SEAttention, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, sequence_length)
            
        Returns:
            weighted_x: 加权后的张量
        """
        batch_size, channels, seq_len = x.size()
        
        # 全局平均池化
        y = self.global_avg_pool(x).view(batch_size, channels)
        
        # 通过全连接层生成注意力权重
        attention_weights = self.fc(y).view(batch_size, channels, 1)
        
        # 应用注意力权重
        weighted_x = x * attention_weights.expand_as(x)
        
        return weighted_x

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, channels, reduction=16):
        """
        初始化通道注意力
        
        Args:
            channels: 输入通道数
            reduction: 降维比例
        """
        super(ChannelAttention, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 最大池化和平均池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, sequence_length)
            
        Returns:
            weighted_x: 加权后的张量
        """
        # 最大池化和平均池化
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        
        # 融合两种池化的结果
        attention_weights = self.sigmoid(max_out + avg_out)
        
        # 应用注意力权重
        weighted_x = x * attention_weights
        
        return weighted_x

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力
        
        Args:
            kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        
        self.kernel_size = kernel_size
        
        # 1D卷积层
        self.conv = nn.Conv1d(
            in_channels=2,  # 最大池化和平均池化的结果
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, sequence_length)
            
        Returns:
            weighted_x: 加权后的张量
        """
        # 在通道维度上进行最大池化和平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # 拼接两种池化的结果
        concat = torch.cat([max_out, avg_out], dim=1)
        
        # 通过1D卷积生成空间注意力权重
        attention_weights = self.sigmoid(self.conv(concat))
        
        # 应用注意力权重
        weighted_x = x * attention_weights
        
        return weighted_x

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        """
        初始化多头自注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑回原始形状
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        output = self.w_o(context)
        
        return output

class MultiScaleAttention(nn.Module):
    """多尺度注意力机制"""
    
    def __init__(self, channels, scales=[1, 3, 5]):
        """
        初始化多尺度注意力
        
        Args:
            channels: 输入通道数
            scales: 多尺度卷积核大小列表（建议使用奇数以保持序列长度）
        """
        super(MultiScaleAttention, self).__init__()
        
        self.channels = channels
        self.scales = scales
        
        # 计算每个尺度的输出通道数
        channels_per_scale = channels // len(scales)
        
        # 不同尺度的卷积层（使用奇数kernel确保padding对称）
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(channels, channels_per_scale, 
                     kernel_size=scale, padding=(scale-1)//2)
            for scale in scales
        ])
        
        # 融合层（输入通道数=实际拼接后的通道数）
        concatenated_channels = channels_per_scale * len(scales)
        self.fusion_conv = nn.Conv1d(concatenated_channels, channels, kernel_size=1)
        
        # 注意力权重生成
        self.attention_conv = nn.Conv1d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, sequence_length)
            
        Returns:
            weighted_x: 加权后的张量
        """
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.conv_layers:
            feature = conv(x)
            multi_scale_features.append(feature)
        
        # 拼接多尺度特征
        concatenated = torch.cat(multi_scale_features, dim=1)
        
        # 融合特征
        fused = self.fusion_conv(concatenated)
        
        # 生成注意力权重
        attention_weights = self.sigmoid(self.attention_conv(fused))
        
        # 应用注意力权重
        weighted_x = x * attention_weights
        
        return weighted_x

class CombinedAttention(nn.Module):
    """组合注意力机制，结合多种注意力"""
    
    def __init__(self, channels, d_model, num_heads=8, reduction=16):
        """
        初始化组合注意力
        
        Args:
            channels: 输入通道数
            d_model: 模型维度
            num_heads: 注意力头数
            reduction: 降维比例
        """
        super(CombinedAttention, self).__init__()
        
        self.channels = channels
        self.d_model = d_model
        
        # 不同的注意力机制
        self.se_attention = SEAttention(channels, reduction)
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        self.multi_head_attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, sequence_length)
            
        Returns:
            output: 输出张量
        """
        # 转换输入格式以适应不同的注意力机制
        # 对于卷积注意力，保持 (batch_size, channels, seq_len)
        # 对于多头注意力，转换为 (batch_size, seq_len, d_model)
        
        # SE注意力
        se_out = self.se_attention(x)
        
        # 通道注意力
        channel_out = self.channel_attention(x)
        
        # 空间注意力
        spatial_out = self.spatial_attention(x)
        
        # 组合卷积注意力结果
        conv_attention_out = se_out + channel_out + spatial_out
        
        # 转换为多头注意力的输入格式
        # 假设 channels = d_model，否则需要投影层
        if self.channels != self.d_model:
            # 这里需要添加投影层，简化处理
            pass
        
        # 转置以适应多头注意力
        multi_head_input = conv_attention_out.transpose(1, 2)
        
        # 多头注意力
        multi_head_out = self.multi_head_attention(multi_head_input)
        
        # 转置回原始格式
        multi_head_out = multi_head_out.transpose(1, 2)
        
        # 融合结果
        # 这里简化处理，实际应用中可能需要更复杂的融合策略
        output = conv_attention_out + multi_head_out
        
        return output

def test_attention_mechanisms():
    """测试注意力机制"""
    print("测试注意力机制...")
    
    # 设置参数
    batch_size = 32
    channels = 128
    sequence_length = 30
    d_model = 128
    
    # 创建测试数据
    x_conv = torch.randn(batch_size, channels, sequence_length)
    x_transformer = torch.randn(batch_size, sequence_length, d_model)
    
    # 测试SE注意力
    print("\n测试SE注意力...")
    se_attention = SEAttention(channels)
    se_out = se_attention(x_conv)
    print(f"输入形状: {x_conv.shape}")
    print(f"输出形状: {se_out.shape}")
    
    # 测试通道注意力
    print("\n测试通道注意力...")
    channel_attention = ChannelAttention(channels)
    channel_out = channel_attention(x_conv)
    print(f"输入形状: {x_conv.shape}")
    print(f"输出形状: {channel_out.shape}")
    
    # 测试空间注意力
    print("\n测试空间注意力...")
    spatial_attention = SpatialAttention()
    spatial_out = spatial_attention(x_conv)
    print(f"输入形状: {x_conv.shape}")
    print(f"输出形状: {spatial_out.shape}")
    
    # 测试多头自注意力
    print("\n测试多头自注意力...")
    multi_head_attention = MultiHeadSelfAttention(d_model)
    multi_head_out = multi_head_attention(x_transformer)
    print(f"输入形状: {x_transformer.shape}")
    print(f"输出形状: {multi_head_out.shape}")
    
    # 测试多尺度注意力
    print("\n测试多尺度注意力...")
    multi_scale_attention = MultiScaleAttention(channels)
    multi_scale_out = multi_scale_attention(x_conv)
    print(f"输入形状: {x_conv.shape}")
    print(f"输出形状: {multi_scale_out.shape}")
    
    print("\n注意力机制测试完成！")

# 测试代码已移至 tests/ 目录

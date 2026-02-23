"""
MABG (Multi-Attention Bidirectional Gated Recurrent Unit) 混合模型实现
结合BGRU和多重注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .bgru import BGRUModel, EnhancedBGRU
from .attention import (
    SEAttention, ChannelAttention, SpatialAttention, 
    MultiHeadSelfAttention, MultiScaleAttention
)

class MABGModel(nn.Module):
    """多重注意力双向门控循环单元模型（单特征版本）"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, 
                 attention_heads=8, reduction=16):
        """
        初始化MABG模型
        
        Args:
            input_size: 输入特征维度（单特征为1）
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
            attention_heads: 注意力头数
            reduction: 注意力降维比例
        """
        super(MABGModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入dropout层（对抗过拟合）
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # BGRU层
        self.bgru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 额外的dropout层（单层GRU需要手动添加）
        self.bgru_dropout = nn.Dropout(dropout)
        
        # 注意力机制
        self.multi_head_attention = MultiHeadSelfAttention(
            hidden_size * 2, attention_heads, dropout
        )
        
        # 池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 输出层（简化版本）
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """
        前向传播（适用于单特征输入）
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, 1)
            
        Returns:
            output: 预测结果，形状为 (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # 输入dropout
        x = self.input_dropout(x)
        
        # 初始化BGRU隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # BGRU前向传播
        bgru_out, _ = self.bgru(x, h0)  # (batch_size, seq_len, hidden_size * 2)
        
        # 应用dropout
        bgru_out = self.bgru_dropout(bgru_out)
        
        # 应用多头自注意力
        att_out = self.multi_head_attention(bgru_out)
        
        # 残差连接
        residual_output = bgru_out + att_out
        
        # 层归一化
        normalized_output = self.layer_norm(residual_output)
        
        # 全局池化
        avg_pooled = self.global_avg_pool(normalized_output.transpose(1, 2)).squeeze(-1)
        max_pooled = self.global_max_pool(normalized_output.transpose(1, 2)).squeeze(-1)
        
        # 拼接池化结果
        pooled_concat = torch.cat([avg_pooled, max_pooled], dim=-1)
        
        # 输出层
        output = self.output_layers(pooled_concat)
        
        return output

class EnhancedMABG(nn.Module):
    """增强版MABG模型，包含更多高级特性"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2,
                 attention_heads=8, reduction=16, use_multi_scale=True):
        """
        初始化增强版MABG模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
            attention_heads: 注意力头数
            reduction: 降维比例
            use_multi_scale: 是否使用多尺度注意力
        """
        super(EnhancedMABG, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_multi_scale = use_multi_scale
        
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
        
        # 注意力机制
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_size * 2, attention_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 多尺度注意力（可选）
        if use_multi_scale:
            self.multi_scale_attention = MultiScaleAttention(hidden_size * 2)
        
        # Dropout层
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 预测结果，形状为 (batch_size, 1)
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 通过多个BGRU层
        for i, (bgru, layer_norm, attention, dropout) in enumerate(
            zip(self.bgru_layers, self.layer_norms, self.attention_layers, self.dropout_layers)
        ):
            # BGRU前向传播
            bgru_out, _ = bgru(x)
            
            # 层归一化
            bgru_out = layer_norm(bgru_out)
            
            # 自注意力机制
            attn_out = attention(bgru_out)
            
            # 残差连接
            x = bgru_out + attn_out
            
            # Dropout
            x = dropout(x)
        
        # 多尺度注意力（可选）
        if self.use_multi_scale:
            # 转换维度
            x_conv = x.transpose(1, 2)
            multi_scale_out = self.multi_scale_attention(x_conv)
            x = multi_scale_out.transpose(1, 2)
        
        # 特征融合
        fused_features = self.feature_fusion(x)
        
        # 全局平均池化
        pooled = torch.mean(fused_features, dim=1)
        
        # 输出投影
        output = self.output_projection(pooled)
        
        return output

class MABGWithTemporalAttention(nn.Module):
    """带时间注意力的MABG模型"""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2,
                 attention_heads=8, temporal_heads=4):
        """
        初始化带时间注意力的MABG模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout概率
            attention_heads: 空间注意力头数
            temporal_heads: 时间注意力头数
        """
        super(MABGWithTemporalAttention, self).__init__()
        
        # BGRU层
        self.bgru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 空间注意力（特征维度）
        self.spatial_attention = MultiHeadSelfAttention(
            hidden_size * 2, attention_heads, dropout
        )
        
        # 时间注意力（序列维度）
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding(1000, hidden_size * 2)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size * 2, 1)
        
    def _create_positional_encoding(self, max_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            output: 预测结果，形状为 (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # BGRU前向传播
        bgru_out, _ = self.bgru(x)
        
        # 添加位置编码
        if seq_len <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
            bgru_out = bgru_out + pos_enc
        
        # 空间注意力
        spatial_out, _ = self.spatial_attention(bgru_out, bgru_out, bgru_out)
        
        # 时间注意力
        temporal_out, _ = self.temporal_attention(bgru_out, bgru_out, bgru_out)
        
        # 融合空间和时间注意力
        fused = torch.cat([spatial_out, temporal_out], dim=-1)
        fused = self.fusion(fused)
        
        # 全局平均池化
        pooled = torch.mean(fused, dim=1)
        
        # 输出层
        output = self.output_layer(pooled)
        
        return output

def test_mabg_models():
    """测试MABG模型"""
    print("测试MABG模型...")
    
    # 设置参数
    batch_size = 32
    sequence_length = 30
    input_size = 7  # 6个百度指数关键词 + 1个ILI病例数
    hidden_size = 64
    num_layers = 2
    
    # 创建测试数据
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # 测试基础MABG模型
    print("\n测试基础MABG模型...")
    model1 = MABGModel(input_size, hidden_size, num_layers)
    output1 = model1(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    print(f"参数数量: {sum(p.numel() for p in model1.parameters())}")
    
    # 测试增强版MABG模型
    print("\n测试增强版MABG模型...")
    model2 = EnhancedMABG(input_size, hidden_size, num_layers)
    output2 = model2(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output2.shape}")
    print(f"参数数量: {sum(p.numel() for p in model2.parameters())}")
    
    # 测试带时间注意力的MABG模型
    print("\n测试带时间注意力的MABG模型...")
    model3 = MABGWithTemporalAttention(input_size, hidden_size, num_layers)
    output3 = model3(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output3.shape}")
    print(f"参数数量: {sum(p.numel() for p in model3.parameters())}")
    
    print("\nMABG模型测试完成！")

# 测试代码已移至 tests/ 目录

"""
Multi-Trend Cross-Attention Model for COVID-19 Prediction

使用交叉注意力机制整合多个Google Trends特征：
- flu_Trends
- COVID_19_Trends  
- fever_Trends
- loss_of_smell_Trends

设计理念：
1. 每个trend独立编码（避免特征冗余）
2. Cross-Attention让trends相互交互（捕获关联模式）
3. 自适应加权融合（自动学习重要性）
"""

import torch
import torch.nn as nn


class MultiTrendCrossAttention(nn.Module):
    """
    多趋势交叉注意力模型
    
    架构：
    Input Trends (4个) -> Independent Encoders -> Cross-Attention -> Fusion -> GRU -> Output
    """
    
    def __init__(
        self, 
        num_trends=4,           # Trends数量
        trend_dim=4,            # 每个trend的维度（1个值+3个lags）
        encoder_hidden=16,      # 每个trend encoder的hidden size
        num_heads=4,            # attention头数
        fusion_hidden=24,       # 融合后的GRU hidden size
        dropout=0.35,
        output_dim=1
    ):
        super(MultiTrendCrossAttention, self).__init__()
        
        self.num_trends = num_trends
        self.trend_dim = trend_dim
        self.encoder_hidden = encoder_hidden
        
        # ============ 1. Trend Independent Encoders ============
        # 每个trend用独立的GRU编码，避免特征混淆
        self.trend_encoders = nn.ModuleList([
            nn.GRU(
                input_size=trend_dim,      # 每个时间步：trend值 + 3个lag
                hidden_size=encoder_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0  # 单层不用dropout
            )
            for _ in range(num_trends)
        ])
        
        # ============ 2. Cross-Attention Layer ============
        # 让不同trends相互关注，捕获关联模式
        # 例如：fever和loss_of_smell可能共同指示COVID严重程度
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=encoder_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention后的LayerNorm（稳定训练）
        self.attn_norm = nn.LayerNorm(encoder_hidden)
        
        # ============ 3. Adaptive Fusion ============
        # 学习每个trend的重要性权重
        self.fusion_weights = nn.Sequential(
            nn.Linear(num_trends * encoder_hidden, num_trends),
            nn.Softmax(dim=1)
        )
        
        # ============ 4. Main GRU Backbone ============
        # 处理融合后的时序特征
        self.fusion_gru = nn.GRU(
            input_size=encoder_hidden,
            hidden_size=fusion_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # ============ 5. Output Layer ============
        self.fc = nn.Linear(fusion_hidden, output_dim)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, total_features)
               total_features = num_trends * trend_dim + additional_features
               例如：(32, 6, 11) = 8(trends) + 3(seasonal)
        
        Returns:
            output: (batch_size, 1) 预测值
            attn_weights: (batch_size, num_trends, num_trends) 注意力权重（用于可视化）
        """
        batch_size, seq_len, total_features = x.shape
        
        # ============ Step 1: 分离trend特征和其他特征 ============
        trend_features_dim = self.num_trends * self.trend_dim
        
        # 提取trend特征和其他特征（如季节特征）
        trend_features = x[:, :, :trend_features_dim]  # (batch, seq, num_trends*trend_dim)
        additional_features = x[:, :, trend_features_dim:]  # (batch, seq, remaining)
        
        # 将trend特征reshape为 (batch, seq, num_trends, trend_dim)
        x_reshaped = trend_features.view(batch_size, seq_len, self.num_trends, self.trend_dim)
        
        encoded_trends = []
        for i in range(self.num_trends):
            # 提取第i个trend: (batch, seq, trend_dim)
            trend_input = x_reshaped[:, :, i, :]
            
            # 通过独立的GRU编码器
            encoded, _ = self.trend_encoders[i](trend_input)  # (batch, seq, encoder_hidden)
            
            # 只保留最后一个时间步的输出
            encoded_trends.append(encoded[:, -1, :])  # (batch, encoder_hidden)
        
        # Stack成 (batch, num_trends, encoder_hidden)
        stacked_trends = torch.stack(encoded_trends, dim=1)
        
        # ============ Step 2: Cross-Attention ============
        # 让trends相互关注，捕获关联模式
        attended, attn_weights = self.cross_attention(
            stacked_trends,  # query
            stacked_trends,  # key
            stacked_trends   # value
        )  # attended: (batch, num_trends, encoder_hidden)
        
        # 残差连接 + LayerNorm
        attended = self.attn_norm(attended + stacked_trends)
        
        # ============ Step 3: Adaptive Fusion ============
        # 学习每个trend的重要性权重
        attended_flat = attended.view(batch_size, -1)  # (batch, num_trends*encoder_hidden)
        fusion_weights = self.fusion_weights(attended_flat)  # (batch, num_trends)
        
        # 加权求和
        weighted_trends = (attended * fusion_weights.unsqueeze(-1)).sum(dim=1)  # (batch, encoder_hidden)
        
        # ============ Step 4: 通过主干GRU ============
        # 添加时间维度 (batch, 1, encoder_hidden)
        weighted_trends = weighted_trends.unsqueeze(1)
        
        gru_out, _ = self.fusion_gru(weighted_trends)  # (batch, 1, fusion_hidden)
        
        # ============ Step 5: 输出预测 ============
        last_output = gru_out[:, -1, :]  # (batch, fusion_hidden)
        dropped = self.dropout(last_output)
        output = self.fc(dropped)  # (batch, 1)
        
        return output, fusion_weights  # 返回权重用于可视化


class SimplifiedMultiTrendAttention(nn.Module):
    """
    简化版多趋势注意力模型（更少参数，适合小数据集）
    
    设计理念：
    - 不用独立encoders，直接对trend特征做attention
    - 减少过拟合风险
    """
    
    def __init__(
        self,
        num_trends=4,
        trend_dim=4,
        hidden_size=24,
        num_heads=2,
        dropout=0.35,
        output_dim=1
    ):
        super(SimplifiedMultiTrendAttention, self).__init__()
        
        self.num_trends = num_trends
        self.trend_dim = trend_dim
        self.embed_dim = trend_dim  # 直接用原始维度
        
        # ============ 1. 简单的特征投影 ============
        self.trend_projections = nn.ModuleList([
            nn.Linear(trend_dim, trend_dim)
            for _ in range(num_trends)
        ])
        
        # ============ 2. Self-Attention ============
        self.self_attention = nn.MultiheadAttention(
            embed_dim=trend_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # ============ 3. 主干GRU ============
        self.gru = nn.GRU(
            input_size=trend_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, total_features)
               total_features = num_trends * trend_dim + additional_features
        
        Returns:
            output: (batch_size, 1)
            attn_weights: (batch_size, num_trends, num_trends)
        """
        batch_size, seq_len, total_features = x.shape
        
        # 分离trend特征
        trend_features_dim = self.num_trends * self.trend_dim
        trend_features = x[:, :, :trend_features_dim]
        
        # 取最后一个时间步
        x_last = trend_features[:, -1, :]  # (batch, num_trends*trend_dim)
        x_reshaped = x_last.view(batch_size, self.num_trends, self.trend_dim)
        
        # 投影每个trend
        projected = []
        for i in range(self.num_trends):
            proj = self.trend_projections[i](x_reshaped[:, i, :])
            projected.append(proj)
        
        projected = torch.stack(projected, dim=1)  # (batch, num_trends, trend_dim)
        
        # Self-Attention
        attended, attn_weights = self.self_attention(
            projected, projected, projected
        )
        
        # 融合（平均）
        fused = attended.mean(dim=1, keepdim=True)  # (batch, 1, trend_dim)
        
        # GRU
        gru_out, _ = self.gru(fused)
        
        # 输出
        output = self.fc(self.dropout(gru_out[:, -1, :]))
        
        return output, attn_weights

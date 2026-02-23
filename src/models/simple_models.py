"""
极简LSTM模型 - 用于测试数据的基本可预测性

这是一个尽可能简单的时间序列预测模型
如果这个模型R²都很低，说明数据本身难以预测
"""

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """极简单层LSTM模型"""
    
    def __init__(self, input_size=1, hidden_size=16, dropout=0.4):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 单层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0  # 单层不用dropout
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 简单的输出层
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, 1)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 只用最后一个时间步
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # 输出
        output = self.fc(dropped)
        
        return output


class SimpleGRU(nn.Module):
    """极简单层GRU模型"""
    
    def __init__(self, input_size=1, hidden_size=16, dropout=0.4):
        super(SimpleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 单层GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """前向传播"""
        # GRU
        gru_out, h_n = self.gru(x)
        
        # 最后时间步
        last_output = gru_out[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # 输出
        output = self.fc(dropped)
        
        return output


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("极简模型测试")
    print("=" * 60)
    
    # 测试SimpleLSTM
    model_lstm = SimpleLSTM(input_size=1, hidden_size=16, dropout=0.4)
    params_lstm = count_parameters(model_lstm)
    
    print(f"\n【SimpleLSTM】")
    print(f"  Hidden Size: 16")
    print(f"  参数量: {params_lstm:,}")
    print(f"  vs MABG优化后: {params_lstm} vs 27,649 (-{(1-params_lstm/27649)*100:.1f}%)")
    
    # 测试前向传播
    dummy_input = torch.randn(8, 30, 1)  # batch=8, seq=30, features=1
    output = model_lstm(dummy_input)
    print(f"  输入形状: {tuple(dummy_input.shape)}")
    print(f"  输出形状: {tuple(output.shape)}")
    print(f"  ✓ 前向传播成功")
    
    # 测试SimpleGRU
    model_gru = SimpleGRU(input_size=1, hidden_size=16, dropout=0.4)
    params_gru = count_parameters(model_gru)
    
    print(f"\n【SimpleGRU】")
    print(f"  Hidden Size: 16")
    print(f"  参数量: {params_gru:,}")
    
    output_gru = model_gru(dummy_input)
    print(f"  ✓ 前向传播成功")
    
    print("\n" + "=" * 60)
    print(f"SimpleLSTM参数: {params_lstm:,}")
    print(f"SimpleGRU参数:  {params_gru:,}")
    print(f"MABG优化后:     27,649")
    print(f"MABG原始:       183,041")
    print(f"\n参数减少: {(1-params_lstm/183041)*100:.1f}% (LSTM)")
    print(f"样本/参数比: 240/{params_lstm} = {240/params_lstm:.2f}")
    print("=" * 60)

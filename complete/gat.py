"""
图注意力网络（GAT）实现
功能：基于构建的图数据训练场景图生成模型
依赖数据：output/graphs/*.npz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    """多头图注意力网络"""
    def __init__(self, node_dim, hidden_dim=256, heads=4, num_layers=2, num_classes=51):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GATConv(node_dim, hidden_dim, heads=heads))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        self.layers2 = nn.ModuleList()
        
        # 输入层
        self.layers2.append(GATConv(node_dim, hidden_dim, heads=heads))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers2.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        # 输出层（边分类）
        # I changed the input dim from 2 to 4, given that we will take from 2 graphs, 
        self.edge_classifier = nn.Linear(hidden_dim * heads * 4, num_classes)  # num_classes是类别数


        # define the beta nad alpha
        alpha = nn.Parameter(torch.tensor(0.5))
        beta = nn.Parameter(torch.tensor(0.5))


    def forward(self, x1, edge_index1, x2, edge_index2):
        # 消息传递
        for layer in self.layers:
            x = F.elu(layer(x1, edge_index1))
        
        # 边特征生成
        src, dst = edge_index1

        for layer in self.layers2:
            x2 = F.elu(layer(x2, edge_index2))
        
        # 边特征生成
        src2, dst2 = edge_index2

        edge_feats = torch.cat([x[src], x[dst], x2[src], x2[dst]], dim=1)
        edge_feats2 = torch.cat([x[src2], x[dst2], x2[src2], x2[dst2]], dim=1)

        # Now given the difference edges between edge_feats and edge_feats2, formula from algorithm 3 line 15, 
        # you should do the operation 
        # 输出 logits

       # Concatenate edge indices and find unique edges
        all_edges = torch.cat([edge_index1, edge_index2], dim=1)
        unique_edges, inverse_indices = torch.unique(all_edges, dim=1, return_inverse=True)

        # Initialize new edge features
        new_edge_feats = torch.zeros((unique_edges.shape[1], edge_index1.shape[1]), device=edge_feats.device)

        # Mapping edges back to original indices
        edge_map = inverse_indices[:, :edge_index1.shape[0]]
        edge_map2 = inverse_indices[:, edge_index2.shape[0]:]

        # Mask for edges that exist in each set
        mask1 = (edge_map[0] < edge_index1.shape[0])
        mask2 = (edge_map2[0] < edge_index2.shape[0])

        # Assign values using efficient indexing
        new_edge_feats[edge_map[0][mask1]] += self.alpha * edge_feats[mask1]
        new_edge_feats[edge_map2[0][mask2]] += self.beta * edge_feats2[mask2]
        new_edge_feats = self.edge_classifier(new_edge_feats)  # 无需sigmoid，cross_entropy内会做softmax

        return torch.nn.functional.softmax(new_edge_feats)


def train_gat(graph_data, num_classes):
    """训练GAT模型"""
    # 加载图数据
    nodes = torch.tensor(graph_data['nodes'], dtype=torch.float32)
    edges = torch.tensor(graph_data['edges'].T, dtype=torch.long)  # [2, E]
    
    # 假设边的标签是类别索引，需要将其转换为整数类型
    edge_labels = torch.tensor(graph_data['weights'], dtype=torch.long)  # 确保是整数类别标签
    
    # 初始化模型
    model = GAT(node_dim=nodes.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(100):
        # 前向传播
        pred = model(nodes, edges)
        
        # 计算损失
        loss = F.cross_entropy(pred, edge_labels)
        
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

if __name__ == "__main__":
    # 示例：加载训练图数据
    import numpy as np
    graph_data = np.load("output/graphs/train_graphs.npz")["G_60"]
    train_gat(graph_data, num_classes=51)  
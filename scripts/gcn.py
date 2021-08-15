import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

from rich import print
from rich.console import Console
from rich.table import Table


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train():
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss


def test():
    model.eval()
    results = []
    for mask in ["train_mask", "val_mask", "test_mask"]:
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        num_correct = (pred[data[mask]] == data.y[data[mask]])
        acc = int(num_correct.sum()) / int(data[mask].sum())
        results.append(acc)
    
    return results



dataset = Planetoid(root="data/Planetoid", name='Cora', transform=NormalizeFeatures())

print(f"Datasets: {dataset}")
print("=======================")
print(f"NUmber of graphs: {len(dataset)}")
print(f"NUmber of features: {dataset.num_features}")
print(f"NUmber of calsses: {dataset.num_classes}")

data = dataset[0]

print(data)
print("=====================================================================")

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


model = GCN(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


table = Table(title="Result of GCN on Cora-dataset")
table.add_column("Epoch")
table.add_column("Training loss")
table.add_column("Validation Accuracy")
table.add_column("Test Accuracy")


best_val_acc = 0
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if val_acc > best_val_acc:
        table.add_row(f"{epoch}", f"{loss:.3f}", f"{val_acc:.3f}", f"{test_acc:.3f}")
        best_val_acc = val_acc


console = Console()
console.print(table)


# command 
"""
$ python gcn.py

"""

# output
"""

Datasets: Cora()
=======================
NUmber of graphs: 1
NUmber of features: 1433
NUmber of calsses: 7
Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], 
val_mask=[2708], x=[2708, 1433], y=[2708])

=====================================================================
Number of nodes: 2708
Number of edges: 10556
Average node degree: 3.90
Number of training nodes: 140
Training node label rate: 0.05
Contains isolated nodes: False
Contains self-loops: False
Is undirected: True


                Result of GCN on Cora-dataset                 
┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Epoch ┃ Training loss ┃ Validation Accuracy ┃ Test Accuracy ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 1     │ 1.945         │ 0.354               │ 0.398         │
│ 2     │ 1.938         │ 0.442               │ 0.448         │
│ 3     │ 1.931         │ 0.476               │ 0.466         │
│ 4     │ 1.923         │ 0.514               │ 0.507         │
│ 5     │ 1.913         │ 0.536               │ 0.558         │
│ 6     │ 1.908         │ 0.582               │ 0.602         │
│ 7     │ 1.892         │ 0.672               │ 0.670         │
│ 8     │ 1.881         │ 0.720               │ 0.731         │
│ 9     │ 1.873         │ 0.732               │ 0.755         │
│ 10    │ 1.862         │ 0.760               │ 0.766         │
│ 12    │ 1.840         │ 0.766               │ 0.766         │
│ 37    │ 1.427         │ 0.768               │ 0.778         │
│ 38    │ 1.364         │ 0.770               │ 0.779         │
│ 40    │ 1.336         │ 0.772               │ 0.784         │
│ 41    │ 1.362         │ 0.774               │ 0.784         │
│ 42    │ 1.287         │ 0.776               │ 0.784         │
│ 50    │ 1.171         │ 0.780               │ 0.798         │
│ 53    │ 1.117         │ 0.782               │ 0.799         │
│ 55    │ 1.059         │ 0.784               │ 0.804         │
│ 57    │ 1.031         │ 0.786               │ 0.803         │
│ 67    │ 0.874         │ 0.788               │ 0.809         │
│ 83    │ 0.755         │ 0.790               │ 0.809         │
│ 84    │ 0.669         │ 0.792               │ 0.810         │
│ 97    │ 0.596         │ 0.794               │ 0.814         │
│ 123   │ 0.497         │ 0.796               │ 0.818         │
│ 136   │ 0.416         │ 0.800               │ 0.819         │
│ 182   │ 0.342         │ 0.802               │ 0.820         │
└───────┴───────────────┴─────────────────────┴───────────────┘

"""

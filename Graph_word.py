import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import Node2Vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


dataset = KarateClub()
data = dataset[0]


model = Node2Vec(edge_index=data.edge_index, embedding_dim=16, walk_length=5, context_size=3, walks_per_node=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_node2vec():
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        batch = torch.arange(data.num_nodes, device=data.edge_index.device)
        pos_rw = model.pos_sample(batch)
        neg_rw = model.neg_sample(batch)
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch no: {epoch+1}, Loss: {loss.item():.3f}")

train_node2vec()

model.eval()
embeddings = model().detach().numpy()

kmeans = KMeans(n_clusters=2, random_state=42)
predicted_labels = kmeans.fit_predict(embeddings)

plt.scatter(embeddings[:, 0], embeddings[:, 1], c=predicted_labels, cmap="Set2", s=100)
plt.title("Node Clusters")
plt.show()

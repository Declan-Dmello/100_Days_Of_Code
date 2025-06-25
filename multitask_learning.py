import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, num_samples=500):
        self.data = torch.rand(num_samples, 5)
        self.task_a_labels = torch.sum(self.data, dim=1)
        self.task_b_labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.task_a_labels[idx], self.task_b_labels[idx]


class SimpleMultitaskModel(nn.Module):
    def __init__(self):
        super(SimpleMultitaskModel, self).__init__()
        self.shared = nn.Linear(5, 10)  # Shared
        self.task_a = nn.Linear(10, 1)  #for regres
        self.task_b = nn.Linear(10, 2)  # for classif

    def forward(self, x):
        shared = torch.relu(self.shared(x))
        task_a_output = self.task_a(shared)  # the Regression output
        task_b_output = self.task_b(shared)  # the Classification output
        return task_a_output, task_b_output

def train():
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleMultitaskModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()


    for epoch in range(5):
        total_loss = 0.0
        for data, task_a_labels, task_b_labels in dataloader:
            optimizer.zero_grad()

            task_a_output, task_b_output = model(data)

            regression_loss = regression_loss_fn(task_a_output.squeeze(), task_a_labels)
            classification_loss = classification_loss_fn(task_b_output, task_b_labels)
            loss = regression_loss + classification_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/5], Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()

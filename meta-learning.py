import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_omniglot_data(root="data/omniglot"):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = datasets.Omniglot(root, background=True, download=True, transform=transform)
    test_dataset = datasets.Omniglot(root, background=False, download=True, transform=transform)

    def group_by_class(dataset):
        data_by_class = defaultdict(list)
        for img, label in dataset:
            data_by_class[label].append(img)
        return data_by_class

    train_data = group_by_class(train_dataset)
    test_data = group_by_class(test_dataset)

    return train_data, test_data



def sample_task(data_by_class, num_classes=5, num_shots=1, num_queries=15):
    classes = random.sample(list(data_by_class.keys()), num_classes)

    label_mapping = {original_label: i for i, original_label in enumerate(classes)}

    support_set = []
    query_set = []

    for cls in classes:
        images = random.sample(data_by_class[cls], num_shots + num_queries)
        support_set.extend((img, label_mapping[cls]) for img in images[:num_shots])
        query_set.extend((img, label_mapping[cls]) for img in images[num_shots:])

    return support_set, query_set


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = None

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)

        if self.fc is None:

            input_size = x.view(x.size(0), -1).shape[1]
            self.fc = nn.Linear(input_size, num_classes).to(x.device)

        x = x.view(x.size(0), -1)
        return self.fc(x)


inner_lr = 0.02
meta_lr = 0.0005
inner_steps = 10
meta_epochs = 50
tasks_per_batch = 16
num_classes = 5
num_shots = 1
num_queries = 15


def maml(meta_model, train_data):
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)

    for epoch in range(meta_epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        for _ in range(tasks_per_batch):

            support_set, query_set = sample_task(train_data, num_classes, num_shots, num_queries)


            support_imgs, support_labels = zip(*support_set)
            query_imgs, query_labels = zip(*query_set)

            support_imgs = torch.stack(support_imgs).to(device)
            support_labels = torch.tensor([label for label in support_labels]).to(device)
            query_imgs = torch.stack(query_imgs).to(device)
            query_labels = torch.tensor([label for label in query_labels]).to(device)


            cloned_model = CNN(num_classes).to(device)
            cloned_model.load_state_dict(meta_model.state_dict())
            task_optimizer = optim.SGD(cloned_model.parameters(), lr=inner_lr)


            for _ in range(inner_steps):
                logits = cloned_model(support_imgs)
                loss = nn.CrossEntropyLoss()(logits, support_labels)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()


            query_logits = cloned_model(query_imgs)
            task_loss = nn.CrossEntropyLoss()(query_logits, query_labels)


            meta_loss += task_loss / tasks_per_batch

        
        meta_loss.backward()
        meta_optimizer.step()

        print(f"Epoch {epoch + 1}/{meta_epochs}, Meta-Loss: {meta_loss.item():.4f}")

    print("Meta-learning complete!")



train_data, test_data = prepare_omniglot_data()
meta_model = CNN(num_classes=num_classes).to(device)
maml(meta_model, train_data)

from torch.utils.data import Dataset, DataLoader


class MNIST(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


mnist = MNIST(train_images_flattened, train_labels)

print(f"{len(train_images_flattened)} {len(mnist)}")

dl = DataLoader(mnist, batch_size=1000, shuffle=True)

for data, labels in dl:
    print(f"{len(data)}")

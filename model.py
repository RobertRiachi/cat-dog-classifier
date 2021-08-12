import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import itertools

from loader import CustomDataset


class CNN(nn.Module):

    def __init__(self, input_channels, num_classes, batch_size):
        super(CNN, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.conv_1 = nn.Conv2d(input_channels, out_channels=16, kernel_size=4, stride=1, padding=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=2)


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_1 = nn.Linear(64 * 7 * 7, 512)
        #self.fc_2 = nn.Linear(4096, 1024)
        self.fc_2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Start 3x56x56
        x = F.relu(self.conv_1(x))  # 16x56x56
        x = self.maxpool(x)

        x = F.relu(self.conv_2(x))  # 32x28x28
        x = self.maxpool(x)

        x = F.relu(self.conv_3(x))  # 64x14x14
        x = self.maxpool(x) # 64x7x7

        '''
        x = F.relu(self.conv_4(x))  # 128x8x8
        x = self.maxpool(x)  # 128x4x4
        '''
        # Flatten to fit shape, start after batch dim
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        #x = self.fc_3(x)

        return x

    def train(self, training_loader, epochs, learning_rate):

        loss = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for batch_idx, (data, target) in tqdm(enumerate(training_loader)):

                #if batch_idx % 100 == 0:
                print(f"Working on batch {batch_idx}/{len(training_loader)} for {epoch}/{epochs} epochs")

                data = data.to(device=device)
                target = target.to(device=device)

                prediction = self.forward(data)

                loss_val = loss(prediction, target)
                optim.zero_grad()
                loss_val.backward()

                optim.step()

    def test(self, test_loader):
        num_total = 0
        num_correct = 0
        num_class_correct = [0 for _ in range(self.num_classes)]
        num_class_total = [0 for _ in range(self.num_classes)]

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device=device)
                target = target.to(device=device)

                prediction = self.forward(data)

                _, result = prediction.max(1)
                num_correct += (result == target).sum()
                num_total += result.size(0)

                for i in range(target.size()[0]):
                    label = target[i]
                    pred = result[i]

                    if label == pred:
                        num_class_correct[label] += 1
                    num_class_total[label] += 1

            print(f"ConvNet Accuracy is {float(num_correct) / float(num_total) * 100:.2f} for {num_total} samples")
            print(f"Cat accuracy: {num_class_correct[0]/num_class_total[0]}, Dog accuracy: {num_class_correct[1]/num_class_total[1]}")

if __name__ == "__main__":
    batch_size = 32
    num_classes = 10
    input_channels = 3
    learning_rate = 1e-3
    num_epochs = 1

    root = "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/train"
    file = "/mnt/c/Users/Robert/Downloads/cats-vs-dogs/train_data.csv"
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((56, 56)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = CustomDataset(root, file, transforms=img_transforms)

    #print(len(data)*.8)

    # 80-20 Train test split
    training_data, test_data = torch.utils.data.random_split(data, [int(len(data) * .8), int(len(data) * .2)])

    # Crash my CPU
    training_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=32, prefetch_factor=2)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=32, prefetch_factor=2)

    print(f"Using device: {device}")

    model = CNN(input_channels, num_classes, batch_size).to(device=device)
    model.train(training_loader, num_epochs, learning_rate)
    model.test(test_loader)

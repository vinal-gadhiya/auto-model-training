import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data_augmentation_script import Augmentation_Data_Basics

obj1 = Augmentation_Data_Basics(0.5)
l1 = obj1.augment_data()

train_loader = DataLoader(l1)

print(next(iter(train_loader)))

# class simple_model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
#         self.line1 = nn.Linear(in_features=100, out_features=1024)
#         self.line2 = nn.Linear(in_features=1024, out_features=100)
#         self.line3 = nn.Linear(in_features=100, out_features=2)

#     def forward(self, x):
#         x = x
#         x = self.conv1(x)
#         x = nn.ReLU(x)
#         x = self.conv2(x)
#         x = nn.ReLU(x)
#         x = self.line1(x)
#         x = nn.ReLU(x)
#         x = self.line2(x)
#         x = nn.ReLU(x)
#         x = self.line3(x)
#         x = nn.Softmax(x)
#         return x


# model = simple_model()

# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(10):
#     for image in train_loader:
#         pred = model(image)
#         loss = nn.CrossEntropyLoss(pred, 1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


# torch.save(model.state_dict, "model_checkpoint.pth.tar")
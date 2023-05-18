from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

def simclr_dataloader(batch_size = 128):
    train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=True)

    test_data = CIFAR10Pair(root='data', train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,drop_last=True)

    memory_data = CIFAR10Pair(root='data', train=True, transform=test_transform, download=True)
    memory_loader = torch.utils.data.DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, memory_loader

def normal_loader(batch_size = 128):
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_data = CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return loader, test_loader

def get_cifar():
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    return test_dataset
    

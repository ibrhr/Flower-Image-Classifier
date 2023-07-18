import argparse
from collections import OrderedDict
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('--save_dir', default='checkpoints', type=str)
parser.add_argument('--arch', default='densenet121',
                    help='The available options are densenet121 and alexnet', type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)

# it's default value varies depending on the chosen architecture, will set it later below
parser.add_argument('--hidden_units', required=False, type=int)

parser.add_argument('--epochs', default=4, type=int)
parser.add_argument('--gpu', action='store_true')


args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate

# set hidden_units to default value depending on the chosen architecture
hidden_units = args.hidden_units
if hidden_units is None:
    if arch == 'alexnet':
        hidden_units = 4096
    elif arch == 'densenet121':
        hidden_units = 512

epochs = args.epochs
gpu = args.gpu

# Setting the device
if gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('GPU is not available, using CPU instead')
        device = torch.device("cpu")
else:
    device = torch.device("cpu")


# Data loading
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(
    224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(
    224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
testing_transforms = validation_transforms

training_dataset = datasets.ImageFolder(
    train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(
    valid_dir, transform=validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

training_loader = torch.utils.data.DataLoader(  # type: ignore
    training_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(  # type: ignore
    validation_dataset, batch_size=64, shuffle=True)
testing_loader = torch.utils.data.DataLoader(  # type: ignore
    testing_dataset, batch_size=64, shuffle=True)

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def prepare_model(arch: str, hidden_units: int):
    fc1_input: int = 0
    if arch == 'alexnet':
        print('Using alexnet')
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        fc1_input = 9216
    elif arch == 'densenet121':
        print('Using densenet121')
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        fc1_input = 1024
    else:
        print('Architecture not supported')
        exit(1)

    for param in model.parameters():
        param.requires_grad = False

    # Create a new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(fc1_input, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.3)),
        ('fc2', nn.Linear(hidden_units, hidden_units//2)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.3)),
        ('fc3', nn.Linear(hidden_units//2, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier  # type: ignore
    return model


def train_model(model, learning_rate, epochs, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=learning_rate)
    model.to(device)

    epochs = 4
    print_every = 10

    for epoch in range(epochs):
        running_loss = 0
        for step, (inputs, labels) in enumerate(training_loader):
            images, labels = inputs.to(device), labels.to(
                device)  # Moving to GPU or CPU

            optimizer.zero_grad()  # Clearing the gradients

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Run validation and testing then print the results
            if step % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0

                model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:

                        images, labels = images.to(device), labels.to(
                            device)  # Moving to GPU or CPU

                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)
                        validation_loss += loss.item()

                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(
                            equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validation_loader):.3f}.. "
                      f"Validation accuracy: {validation_accuracy/len(validation_loader)*100:.3f}%")

                running_loss = 0

                model.train()
    return model


def main():
    model = prepare_model(arch, hidden_units)
    model = train_model(model, learning_rate, epochs, device)

    model.to('cpu')

    model.class_to_idx = training_dataset.class_to_idx  # type: ignore

    checkpoint = {'classifier': model.classifier,
                  'arch':       arch,
                  'epochs':     epochs,
                  'lr': learning_rate,
                  'state_dict': model.state_dict(),
                  'mapping':    model.class_to_idx
                  }

    torch.save(checkpoint, 'checkpoint.pth')


if __name__ == "__main__":
    main()

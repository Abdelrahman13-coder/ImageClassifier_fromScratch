import argparse
import torch
from torch import nn,optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data_dir',default='flowers', type=str)
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='train_checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16','vgg19_bn','densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden', action='store', dest='hidden', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def load_model(arch, hidden, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        samples = 25088
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
        samples = 1024
    if arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        samples = 4096
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(samples, hidden)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden, hidden)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(hidden, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model = model.to('cuda')
    return model, device ,samples


def train_model(epochs, train_dataloader, test_dataloader, model, device, criterion, optimizer):
    steps = 0
    running_loss = 0
    print_every = 5

    start = time.time()
    print('Model is Training...')

    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(test_dataloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
                running_loss = 0
                model.train()

    end = time.time()
    total_time = end - start
    print(" Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))


def save_checkpoint(file_path, model, train_image_dataset, epochs, optimizer, learning_rate, input_size, output_size, arch, hidden):
    model.class_to_idx = train_image_dataset.class_to_idx
    torch.save({'input_size': input_size,
                'output_size': output_size,
                'structure': model,
                'learning_rate':learning_rate,
                'classifier': model.classifier,
                'epochs': epochs,
                'hidden': hidden,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, 'checkpoint.pth')

    print("The Model is saved...")

def main():
    print("Loading data...")
    args = parse_args()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
 
    train_image_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_image_dataset = datasets.ImageFolder(valid_dir, transform = val_transforms)
    test_image_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(val_image_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=64, shuffle=False)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # getting model, device object and number of input features
    model, device,samples= load_model(args.arch, args.hidden, args.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(args.epochs, train_dataloader, valid_dataloader, model, device, criterion, optimizer)

    file_path = args.save_dir

    output_size = 102
    save_checkpoint(file_path, model, train_image_dataset, args.epochs, optimizer, args.learning_rate,samples,output_size, args.arch,                                   args.hidden)


if __name__ == "__main__":
    main()
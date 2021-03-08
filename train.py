import argparse
import gc
import os
import sys
import time
import uuid

import torch
from torch import nn
from torchvision import transforms, datasets

from workspace_utils import keep_awake

from network import Network

BATCH_SIZE = 16

device = None
criterion = None

train_transforms = None
validation_transforms = None

train_data = None
validation_data = None

train_loader = None
validation_loader = None


def initializeDevice(isGPU):
    # Use GPI if it's available
    if isGPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device [{device}]")

    return device


def trainModel(model, train_loader, validation_loader, criterion, device):
    print("Starting training the neural network ...")
    # Train the network
    steps = 0
    running_loss = 0
    print_every = 5
    start_train = time.time()
    model.to(device)

    for epoch in keep_awake(range(model.no_epochs)):
        start_epoch = time.time()
        for batch_no, (inputs, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()
            gc.collect()
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            model.optimizer.zero_grad()
            start = time.time()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()

            inputs, labels = inputs.to(torch.device('cpu')), labels.to(torch.device('cpu'))

            torch.cuda.empty_cache()
            gc.collect()
            model.optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(
                    f"Epoch {epoch + 1}/{model.no_epochs}.. "
                    f"Train loss: {running_loss / print_every:.3f} "
                    f"Validation loss: {validation_loss / len(validation_loader):.3f} "
                    f"Validation accuracy: {accuracy / len(validation_loader):.3f} "
                )
                running_loss = 0
                model.train()
                print(
                    f"Device = {device}; "
                    f"Time for batch no: {batch_no}/{len(train_loader)}: Avg[{(time.time() - start) / (batch_no + 1):.1f} seconds]")

        print(f"Epoch: {epoch + 1}/{model.no_epochs} Time for epoch: {(time.time() - start_epoch) / 60:.1f} minutes")

    print("Network training completed.")
    print(f"Total training time: {(time.time() - start_train) / 60:.1f} minutes")


def initializeTransforms():
    def normalize_transform():
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize_transform()
    ])
    print("Data transforms created.")

    return train_transforms, validation_transforms


def initializeDataLoader(data_directory, train_transforms, validation_transforms):
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_directory + '/valid', transform=validation_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    print("Data loaders created.")

    return train_data, validation_data, train_loader, validation_loader


def initializeCritrion():
    criterion = nn.NLLLoss()

    return criterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a machine learning model")

    default_model_architecture = 'vgg16'
    default_learning_rate = 0.05
    default_dropout_percentage = 0.02
    default_no_inputs = 25088
    default_no_outputs = 102
    default_layer_sizes = [4096, 256]
    default_epochs = 20
    default_isGPU = False

    parser.add_argument('data_directory', action="store", help="Set directory for data")
    parser.add_argument('-s', '--save_dir', action="store", dest="checkpoints_directory",
                        help="Set directory to save checkpoints")
    parser.add_argument('-a', '--arch', action="store", default=default_model_architecture, dest="model_architecture",
                        help=f"Machine learning architecture to use (default: {default_model_architecture})")
    parser.add_argument('-l', '--learning_rate', action="store", default=default_learning_rate, dest="learning_rate",
                        help=f"Set learning rate (default: {default_learning_rate})", type=float)
    parser.add_argument('-d', '--dropout_percentage', action="store", default=default_dropout_percentage,
                        dest="dropout_percentage",
                        help=f"Set dropout percentage (default: {default_dropout_percentage})", type=float)
    parser.add_argument('-i', '--no_inputs', action="store", default=default_no_inputs, dest="no_inputs",
                        help=f"Set number of inputs to the model (default: {default_no_inputs})", type=int,
                        required=True)
    parser.add_argument('-o', '--no_outputs', action="store", default=default_no_outputs, dest="no_outputs",
                        help=f"Set number of outputs from the model (default: {default_no_outputs})", type=int,
                        required=True)
    parser.add_argument('-u', '--layer_size', action="append", default=default_layer_sizes, dest="layer_sizes",
                        help=f"Specify layer sizes (default: {default_layer_sizes})", type=int)
    parser.add_argument('-e', '--epochs', action="store", default=default_epochs, dest="epochs",
                        help=f"Set number of Training Epochs (default: {default_epochs})", type=int)
    parser.add_argument('-g', '--gpu', action="store_true", default=default_isGPU, dest="isGPU",
                        help=f"Use GPU if available for training (default: {default_isGPU})")
    parser.add_argument('-v', '--version', action="version", version='%(prog)s 1.0',
                        help="Displays the version of the program")

    results = parser.parse_args()
    print(f"Training with the following parameters:{results}")

    if results.checkpoints_directory and not os.path.isdir(results.checkpoints_directory):
        print(f"Checkpoint directory [{results.checkpoints_directory}] not found.")
        sys.exit(-1)
    else:
        checkpoint_file = os.getcwd()
        print(f"Saving check point file in folder [{checkpoint_file}] ...")

    model = Network(
        model_architecture=results.model_architecture,
        no_model_inputs=results.no_inputs,
        no_model_outputs=results.no_outputs,
        layers_input_sizes=sorted(results.layer_sizes, reverse=True),
        learning_rate=results.learning_rate,
        drop_out_percentage=results.dropout_percentage,
        no_epochs=results.epochs
    )
    print(f"Initialized network: {model}")

    device = initializeDevice(isGPU=results.isGPU)

    train_transforms, validation_transforms = initializeTransforms()
    criterion = initializeCritrion()

    train_data, validation_data, train_loader, validation_loader = initializeDataLoader(
        data_directory=results.data_directory,
        train_transforms=train_transforms,
        validation_transforms=validation_transforms)

    trainModel(model=model, train_loader=train_loader, validation_loader=validation_loader, criterion=criterion,
               device=device)

    model.train_data_class_to_idx = train_data.class_to_idx

    checkpoint_file = f"{checkpoint_file}/network_{uuid.uuid4().hex}.pth"
    model.save_checkpoint(filepath=checkpoint_file)

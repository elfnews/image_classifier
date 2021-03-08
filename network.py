from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models


# Network definition
class Network(nn.Module):
    def __init__(self,
                 model_architecture,
                 no_model_inputs,
                 no_model_outputs,
                 layers_input_sizes,
                 learning_rate,
                 drop_out_percentage=0.2,
                 no_epochs=5,
                 train_data_class_to_idx=None):
        '''
        Construct a pretrained network with a new classifier

        :param model_architecture: the pretrained model to use
        :param no_model_inputs: Number of inputs to the model
        :param no_model_outputs: Number of outputs from the model
        :param layers_input_sizes: Sizes inner layers in the network
        :param learning_rate: Learning rate to use while training the network
        :param drop_out_percentage: Drop percentage at each layer with in the network
        :param no_epochs: Number of epochs to use while training the network
        :param train_data_class_to_idx:
        '''
        super().__init__()
        self.model_architecture = model_architecture
        self.no_model_inputs = no_model_inputs
        self.no_model_outputs = no_model_outputs
        self.layers_input_sizes = layers_input_sizes
        self.learning_rate = learning_rate
        self.drop_out_percentage = drop_out_percentage
        self.no_epochs = no_epochs
        self.train_data_class_to_idx = train_data_class_to_idx

        print(f"Creating network with "
              f"architecture[{model_architecture}] "
              f"inputs[{no_model_inputs}] "
              f"outputs[{no_model_outputs}] "
              f"layer inputs[{layers_input_sizes}] "
              f"learning rate[{learning_rate}] "
              f"drop out percentage[{drop_out_percentage}] "
              f"epochs[{no_epochs}] "
              f"class index[{train_data_class_to_idx}]")

        model_method = getattr(models, self.model_architecture)
        print(f"model_method[{model_method}]")
        self.model_delegate = model_method()
        # Freeze parameters so we don't back propagate the pre-trained network
        for param in self.model_delegate.parameters():
            param.requires_grad = False

        # Initialize input to first hidden layer
        layers = nn.ModuleList([nn.Linear(self.no_model_inputs, self.layers_input_sizes[0])])

        # Initialize inner hidden layers
        all_layers_except_output = self.layers_input_sizes[:-1]
        all_layers_except_input = self.layers_input_sizes[1:]
        layer_dimensions = zip(all_layers_except_output,
                               all_layers_except_input)
        layers.extend(
            [
                nn.Linear(no_layer_inputs, no_layer_outputs)
                for no_layer_inputs, no_layer_outputs in layer_dimensions
            ])
        output_layer = nn.Linear(self.layers_input_sizes[-1], self.no_model_outputs)

        modules = []
        for layer_no, layer in enumerate(layers):
            modules.append((f"fc{layer_no}", layer))
            modules.append((f"relu{layer_no}", nn.ReLU(inplace=True)))
            modules.append((f"do{layer_no}", nn.Dropout(p=self.drop_out_percentage)))

        modules.append(("fc_output", output_layer))
        modules.append(("output", nn.LogSoftmax(dim=1)))

        self.classifier = self.model_delegate.classifier = nn.Sequential(OrderedDict(modules))
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def forward(self, x):
        '''
        Forward pass definition. Returns output logits

        :param x: Data to run through the forward pass
        :return: Output logits
        '''
        return self.model_delegate.forward(x)

    def to(self, device):
        super().to(device)
        return self.model_delegate.to(device)

    def eval(self):
        super().eval()
        self.model_delegate.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.model_delegate.train(mode=mode)
        return self

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_architecture': self.model_architecture,
            'no_model_inputs': self.no_model_inputs,
            'no_model_outputs': self.no_model_outputs,
            'layers_input_sizes': self.layers_input_sizes,
            'learning_rate': self.learning_rate,
            'drop_out_percentage': self.drop_out_percentage,
            'no_epochs': self.no_epochs,
            'train_data_class_to_idx': self.train_data_class_to_idx,
            'state_dict': self.model_delegate.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        print(f"Saving checking to [{filepath}] ...")
        torch.save(checkpoint, filepath)
        print("Done.")
        return self

    def load_checkpoint(filepath):
        print(f"Loading checkpoint from [{filepath}] ...")
        checkpoint = torch.load(filepath)
        network_model = Network(
            model_architecture=checkpoint['model_architecture'],
            no_model_inputs=checkpoint['no_model_inputs'],
            no_model_outputs=checkpoint['no_model_outputs'],
            layers_input_sizes=checkpoint['layers_input_sizes'],
            learning_rate=checkpoint['learning_rate'],
            drop_out_percentage=checkpoint['drop_out_percentage'],
            no_epochs=checkpoint['no_epochs'],
            train_data_class_to_idx=checkpoint['train_data_class_to_idx']
        )
        network_model.model_delegate.load_state_dict(checkpoint['state_dict'])
        network_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Done.")
        return network_model

# Pytorch Model class
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary
from torchvision import models
import numpy as np


class SimpleRelationExtractionNet(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim):
        super(SimpleRelationExtractionNet, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(2 + hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, sentences, entities):
        embeds = self.embedding(sentences)
        _, hidden = self.gru(embeds)

        x = torch.cat((entities, hidden.squeeze(0)), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RelationModels(nn.Module):
    DEFAULT_CONFIG = {
        "nn_optimizer": {
            "epochs": 5,
            "batch_size": 64,
            "shuffle": True,
            "optimizer_class": "Adam",
            "learning_rate": 0.01,
            "loss": "binary_crossentropy"
        },
        "nn_model": {
            "input_shape": (63,),
            "layers": [
                {
                    "type": "Linear",
                    "units": 64,
                    "activation": "ReLU"
                },
                {
                    "type": "Linear",
                    "units": 64,
                    "activation": "ReLU"
                },
                {
                    "type": "Linear",
                    "units": 2,
                    "activation": "Sigmoid"
                },
            ]
        }
    }

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        hp=None,
    ):
        super(RelationModels, self).__init__()
        if hp is None:
            hp = self.DEFAULT_CONFIG
        self.hp = hp
        self.nn = self.hp['nn_model']
        self.model = self.SimpleRelationExtractionNet(embedding_dim,
                                                      hidden_dim, vocab_size,
                                                      output_dim)
        self.optimizer = self.create_optimizer()
        self.loss_function = nn.BCELoss() if self.nn_optimizer[
            'loss'] == "binary_crossentropy" else nn.CrossEntropyLoss()

    def create_model(self) -> nn.Module:
        layers = []
        for layer in self.nn['layers']:
            layer_type = layer['type']
            if layer_type == "Linear":
                layers.append(
                    nn.Linear(in_features=layer['units'],
                              out_features=layer['units']))
                if layer['activation'] == "ReLU":
                    layers.append(nn.ReLU())
                elif layer['activation'] == "Sigmoid":
                    layers.append(nn.Sigmoid())
                elif layer['activation'] == "Softmax":
                    layers.append(nn.Softmax())
            else:
                raise NameError(f"Layer type {layer_type} not supported")
        return nn.Sequential(*layers)

    def create_optimizer(self) -> optim.Optimizer:
        if self.nn_optimizer['optimizer_class'] == 'Adam':
            return optim.Adam(self.model.parameters(),
                              lr=self.nn_optimizer['learning_rate'])

    def forward(self, sentences, entities) -> torch.Tensor:
        return self.model(sentences, entities)

    def fit(self, dataloader: torch.utils.data.DataLoader):
        for epoch in range(self.nn_optimizer['epochs']):
            for sentences, entities, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.forward(sentences, entities)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

        print(summary(self))

    def save(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def load(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))

    def predict(self, input_array: torch.Tensor):
        with torch.no_grad():
            input_tensor = torch.from_numpy(np.array(input_array)).float()
            probabilities = self.forward(input_tensor)

            if self.nn['layers'][-1]['activations'] == 'Sigmoid':
                predicted_classes = [
                    1 if prob > 0.5 else 0 for prob in probabilities
                ]
            else:
                predicted_classes = torch.argmax(probabilities, dim=1)

            return probabilities, predicted_classes

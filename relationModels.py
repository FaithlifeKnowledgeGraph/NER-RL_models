import torch 
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class RelationModels(nn.Module):
    DEFAULT_CONFIG = {
        "nn_optimizer": {
            "epochs": 2,
            "batch_size": 128,
            "shuffle": True,
            "optimizer_class": "Adam",
            "learning_rate": 0.01,
            "loss": "binary_crossentropy"
        },
    }
    def __init__(self, model, args=None):
        super(RelationModels, self).__init__()
        if args is None:
            args = self.DEFAULT_CONFIG
        self.model = model
        self.nn_optimizer = args['nn_optimizer']
        self.optimizer = self.create_optimizer()
        self.loss_function = nn.BCELoss() if self.nn_optimizer['loss'] == "binary_crossentropy" else nn.CrossEntropyLoss()

        if args['torch']['device'] == 'cuda':
            torch.cuda.empty_cache()
            if not torch.cuda.is_available():
                print('No CUDA found!')
                exit(-1)
            self._model_device = 'cuda'
            self.model.cuda()
            print(f"Running on GPU: {torch.cuda.device_count()} GPUs")
            if torch.cuda.device_count() > 1:
                self.bert_model = torch.nn.DataParallel(self.bert_model)
        else:
            self._model_device = 'cpu'
            print("Running on CPU")

    def create_optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.nn_optimizer['learning_rate'])
    
    def forward(self, sentences):
        return self.model(sentences.to(self._model_device))

    def fit(self, dataloader: torch.utils.data.DataLoader):
        for epoch in range(self.nn_optimizer['epochs']):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch.to(self._model_device))
                loss = self.loss_function(y_pred.to(self._model_device), 
                    y_batch.to(self._model_device).float().view(-1, 1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.nn_optimizer['epochs']} - Loss: {total_loss/len(dataloader)}")

    def evaluate(self, test_loader: torch.utils.data.DataLoader):
        self.model.eval()
        total_loss = 0
        y_pred = []
        for X_batch, y_batch in test_loader:
            y_pred.append(self.model(X_batch.to(self._model_device)))
            print(f"Test loss: {total_loss/len(test_loader)}")
        return None, torch.cat(y_pred, dim=0)


class SimpleRelationExtractionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(SimpleRelationExtractionModel, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True)

        self.linear = nn.Linear(2 * hidden_dim, 1)
    
    def forward(self, x):
        x = self.word_embeds(x)

        output, (hidden, cell) = self.lstm(x)

        # Take final hidden state from forward and backward LSTM
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        output = self.linear(final_hidden)

        return torch.sigmoid(output)


class RelationExtractionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, pos_embed_size, 
                 hidden_size, num_layers):
        super(RelationExtractionModel, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embed_size)

        self.pos1_embeds = nn.Embedding(MAX_SENTENCE_LENGTH, pos_embed_size)
        self.pos2_embds = nn.Embedding(MAX_SENTENCE_LENGTH, pos_embed_size)

        self.lstm = nn.LSTM(embed_size + 2 * pos_embed_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True)

        self.attention = nn.Linear(2 * hidden_size, 1)

        self.classifier = nn.Linear(2 * hidden_size, 2)
    
    def forward(self, x, pos1, pos2):
        x = self.word_embeds(x)

        pos1 = self.pos1_embeds(pos1)
        pos2 = self.pos2_embeds(pos2)

        # combine word and position embeddings
        x  = torch.cat((x, pos1, pos2), 2)

        x, _ = self.lstm(x)

        attention_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(attention_weights * x, dim=1)

        x = self.classifier(x)

        return torch.sigmoid(x)
    

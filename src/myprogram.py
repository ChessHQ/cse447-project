#!/usr/bin/env python
import os
import string
import random
import json
from sysconfig import get_paths
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
import torch
import torch.nn as nn

# Following tutorial: https://www.youtube.com/watch?v=Mqo7FwgWZFQ
# Also this: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

class MyModel(nn.Module):
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, vocab_size, embed_size=128, hidden_size=512, num_layers=2):
        super(MyModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.to(self.device)

        self.char2idx = {}
        self.idx2char = {}

    def forward(self, x, hidden=None):
        x = x.to(self.device)
        embedded = self.embed(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    @classmethod
    def load_training_data(cls):
        # your code here
        data = []

        ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", token="", encoding='utf-8')
        translator = str.maketrans('', '', string.punctuation)
        for line in ds["dev"]["text"]:
            line = line.strip()
            line = line.translate(translator)
            line = line.replace(" ", "")
            if line:
                data.append(line)

        print(f'Loaded {len(data)} lines from HuggingFace')
        return data
    
    
    @classmethod
    def get_batches(cls, data, batch_size):
        n_batches = len(data) // (batch_size * 100)
        data = data[:n_batches * batch_size * 100]
        x = np.array(data)
        y = np.roll(x, -1)
        x = x.reshape(batch_size, -1)
        y = y.reshape(batch_size, -1)
        return x, y
    

    @classmethod
    def load_test_data(cls, fname):
        with open(fname) as f:
            return [line.strip() for line in f]

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write(f"{p}\n")

    def run_train(self, data: list[str], work_dir):
        # your code here
        text = ''.join(data)
        chars = sorted(list(set(text)))

        self.char2idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        encoded = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)

        batch_size = 64
        seq_length = 100
        epochs = 10

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            total_chars = 0
           
            for i in range(0, len(encoded) - seq_length, seq_length):
               inputs = encoded[i:i+seq_length].unsqueeze(0).to(self.device)
               targets = encoded[i+1:i+seq_length+1].unsqueeze(0).to(self.device)

               optimizer.zero_grad()

               outputs, _ = self(inputs)

               loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))

               loss.backward()
               torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
               optimizer.step()

               total_loss += loss.item() * targets.numel()
               total_chars += targets.numel()

            mean_loss = total_loss / total_chars            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}') 
    
        self.save(work_dir)
        print(f'Model saved to {work_dir}')

    def run_pred(self, data):
        # your code here
        preds = []
        
        for line in data:
            preds.append(self.generate_text(line, 3))
        
        return preds

    def save(self, work_dir):
        # your code here
        model_path = os.path.join(work_dir, 'model.checkpoint')
        torch.save({
            'state_dict': self.state_dict(),
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }, model_path)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model_path = os.path.join(work_dir, 'model.checkpoint')
        checkpoint = torch.load(model_path, map_location='cpu')

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_size=checkpoint['embed_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )

        model.load_state_dict(checkpoint['state_dict'])

        model.char2idx = checkpoint['char2idx']
        model.idx2char = checkpoint['idx2char']

        model.eval()

        model.to(model.device)

        return model

    
    def generate_text(self, prefix, length):
        self.eval()
        hidden = None

        output = None

        if prefix:
            for ch in prefix:
                if ch not in self.char2idx:
                    continue
                inp = torch.tensor([[self.char2idx[ch]]], dtype=torch.long).to(self.device)
                output, hidden = self(inp, hidden)
            if output is None:
                inp = torch.tensor([[random.choice(list(self.char2idx.values()))]], dtype=torch.long).to(self.device)
                output, hidden = self(inp, hidden)
        else:
            inp = torch.tensor([[random.choice(list(self.char2idx.values()))]], dtype=torch.long).to(self.device)
            output, hidden = self(inp, hidden)
        logits = output[:, -1, :]
        topk_indices = torch.topk(logits, k=length).indices[0].tolist()
        topk_chars = [self.idx2char[idx] for idx in topk_indices]
        
        return ''.join(topk_chars)

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Loading training data')
        train_data = MyModel.load_training_data()

        text = ''.join(train_data)
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        print('Instatiating model')
        model = MyModel(vocab_size=vocab_size, embed_size=128, hidden_size=384, num_layers=1)

        print('Training')
        model.run_train(train_data, args.work_dir)

        print('Saving model')
        model.save(args.work_dir)

    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))

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
    def __init__(self):
        super(MyModel, self).__init__()
        self.embed = None
        self.rnn = None
        self.fc = None
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.embed_size = 0
        self.hidden_size = 0
        self.num_layers = 0

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

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
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data: list[str], work_dir):
        # your code here
        text = ''.join(data)
        chars = sorted(list(set(text)))
        char2idx = {ch: idx for idx, ch in enumerate(chars)}
        idx2char = {idx: ch for idx, ch in enumerate(chars)}
        self.char2idx = char2idx
        self.idx2char = idx2char
        data = [char2idx[ch] for ch in text] 

        vocab_size = len(chars)
        embed_size = 128
        hidden_size = 256
        num_layers = 2
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        batch_size = 64
        seq_length = 100
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        epochs = 10

        for epoch in range(epochs):
            x, y = self.get_batches(data, batch_size)
            hidden = None
            for i in range(0, x.shape[1], seq_length):
                inputs = torch.tensor(x[:, i:i+seq_length], dtype=torch.long)      
                targets = torch.tensor(y[:, i:i+seq_length], dtype=torch.long)
                optimizer.zero_grad()

                if hidden is not None:
                    hidden = hidden.detach()
                outputs, hidden = self(inputs, hidden)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, loss: {loss.item():.5f}')
       

    def run_pred(self, data):
        # your code here
        preds = []
        

        for line in data:
            preds.append(self.generate_text(line, 3, None))
        
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
        model = cls()
        model_path = os.path.join(work_dir, 'model.checkpoint')
        checkpoint = torch.load(model_path, map_location='cpu')

        model = cls()

        model.char2idx = checkpoint['char2idx']
        model.idx2char = checkpoint['idx2char']
        model.vocab_size = checkpoint['vocab_size']
        model.embed_size = checkpoint['embed_size']
        model.hidden_size = checkpoint['hidden_size']
        model.num_layers = checkpoint['num_layers']
        model.embed = nn.Embedding(model.vocab_size, model.embed_size)
        model.rnn = nn.RNN(model.embed_size, model.hidden_size, model.num_layers, batch_first=True)
        model.fc = nn.Linear(model.hidden_size, model.vocab_size)

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        return model

    
    def generate_text(self, prefix, length, hidden):
        self.eval()

        if hidden is None:
            hidden = torch.zeros(self.num_layers, 1, self.hidden_size)

        generated = ''

        for char in prefix:
            if char not in self.char2idx:
                continue
            input = torch.tensor([[self.char2idx[char]]], dtype=torch.long)
            output, hidden = self(input, hidden)

        last_char = prefix[-1]
        if last_char not in self.char2idx:
            last_char = random.choice(list(self.char2idx.keys()))
        input = torch.tensor([[self.char2idx[last_char]]], dtype=torch.long)

        for _ in range(length):
            output, hidden = self(input, hidden)
            prob = nn.functional.softmax(output[-1], dim=-1).data
            char_idx = torch.multinomial(prob, 1).item()

            generated += self.idx2char[char_idx]
            input = torch.tensor([[char_idx]], dtype= torch.long)
        return generated


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
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
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

#!/usr/bin/env python
import os
import string
import random
import json
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self):
        self.unigram_probs = {}

    @classmethod
    def load_training_data(cls):
        # your code here
        data = []
        shakespeare_path = os.path.join(os.path.dirname(__file__), '.', 'shakespeare.txt')
    
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(line)
        # For debugging
        # print(f'Lines: {data[:5]}\n')
        # print(f'Words: {words[:5]}\n')

        print(f'Loaded {len(data)} lines from {shakespeare_path}')
        return data


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
        unigram_probs: dict[str, float] = {}
        
        ''' 
        Reason why we use lower case is to reduce the number of unique chars
        For example, if there is a scenario where 'A' and 'a' both have high probability,
        the model would return both of them out of the 3 guesses, which is not ideal.
        '''
        for sentence in data:
            sentence: str
            for char in sentence:
                lower_char = char.lower()
                if lower_char == ' ':
                    continue
                if lower_char in unigram_probs:
                    unigram_probs[lower_char] += 1
                else:
                    unigram_probs[lower_char] = 1
        
        total_chars = sum(unigram_probs.values())
        for char in unigram_probs:
            unigram_probs[char] /= total_chars
        self.unigram_probs = unigram_probs
        

    def run_pred(self, data):
        # your code here
        preds = []
        chars = list(self.unigram_probs.keys())
        probs = list(self.unigram_probs.values())
        
        for inp in data:
            sampled_chars = set()
            i = 0
            while i < 3:
                sampled_char = np.random.choice(chars, p=probs)
                sampled_chars.add(sampled_char)
                i += 1
            preds.append(''.join(sampled_chars))
        
        return preds

    def save(self, work_dir):
        # your code here
        model_path = os.path.join(work_dir, 'model.checkpoint')
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(self.unigram_probs, f, ensure_ascii=False, indent=2)
        print(f'Saved model with {len(self.unigram_probs)} characters to {model_path}')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model = cls()
        model_path = os.path.join(work_dir, 'model.checkpoint')
        with open(model_path, 'r', encoding='utf-8') as f:
            model.unigram_probs = json.load(f)
        print(f'Loaded model with {len(model.unigram_probs)} characters from {model_path}')
        return model


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

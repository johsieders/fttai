# %%
## to be copied from cola.ipynb


# from https://mccormickml.com/2019/07/22/BERT-fine-tuning
# The Corpus of Linguistic Acceptability (CoLA)

# Johannes Siedersleben
# QAware GmbH, Munich
# 10.2.2021

# %%

# uncomment this if import fails
# !pip install wget

import os
import zipfile

import wget

# %%

# download and unzip raw data
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

zipped_file = 'cola_public_1.1.zip'
zipped_dir = './cola_public_1.1/'
unzipped_file = './cola_public/raw/in_domain_train.tsv'

if not os.path.exists(zipped_file):
    wget.download(url, zipped_file)
print('download successful')

if not os.path.exists(zipped_dir):
    zip = zipfile.ZipFile(zipped_file)
    zip.extractall()

print('unzipped file now at ' + unzipped_file)

# %%

# uncomment this if import fails
# !pip install transformers

# Python imports
import random
import pickle
from collections.abc import Callable
from time import perf_counter

# utilities for download and file import
import pandas as pd

# neural metworks support: torch, Huggingface transformers
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# %%

class Logger(object):
    def __init__(self):
        self.protocol = []
        self.counter = 0
        self.char_counter = 0

    def log(self, input: any) -> None:
        print(self.counter, end='')  # I am working
        self.counter = (self.counter + 1) % 10
        self.char_counter = (self.char_counter + 1) % 80
        if self.char_counter == 0:
            print()
        self.protocol.append((perf_counter(), input))

    def getProtocol(self) -> list:
        return self.protocol


# %%

class Learner(object):
    def __init__(self, module: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 schedulerFactory: Callable,
                 device: torch.device):
        self.module = module
        self.optimizer = optimizer
        self.schedulerFactory = schedulerFactory
        self.scheduler = None
        self.device = device

    def train(self, dataloader: DataLoader, logger: Logger) -> None:
        """
        @param dataloader: a dataloader with input_ids at 0, attention_mask at 1, labels at 2
        @param logger: a logger
        @return: None
        This is one epoch. the essential loop of artificial intelligence.
        It runs over all training sentences, one minibatch at a time.
        One loop takes about 2 seconds on a i7 cpu!
        """
        self.module.train()

        for batch in dataloader:
            loss = self.module.forward(input_ids=batch[0],
                                       token_type_ids=None,
                                       attention_mask=batch[1],
                                       labels=batch[2]).loss
            logger.log(loss.item())
            loss.backward()  # compute gradient
            clip_grad_norm_(module.parameters(), 1.0)  # normalize gradient
            self.optimizer.step()  # do one optimization step
            self.scheduler.step()  # adjust learning rate
            self.optimizer.zero_grad()  # reset gradient

    def predict(self, dataloader: DataLoader) -> tuple:
        """
        @param dataloader: a dataloader with input_ids at 0, attention_mask at 1, labels at 2
        @return: tuple of (label, prediction), two tensors
        """
        self.module.eval()
        labels = torch.tensor((), dtype=torch.int, device=self.device)
        predictions = torch.tensor((), dtype=torch.int, device=self.device)

        for batch in dataloader:
            with torch.no_grad():
                logits = self.module.forward(input_ids=batch[0],
                                             token_type_ids=None,
                                             attention_mask=batch[1],
                                             labels=batch[2]).logits
            preds = torch.argmax(logits, dim=1)
            labels = torch.cat((labels, batch[2]))  # collect labels
            predictions = torch.cat((predictions, preds))  # collect predictions

        return labels, predictions

    def fit(self, dataloader: DataLoader, steps_per_epoch, n_epochs: int) -> list:
        """
        @param dataloader: a dataloader with input_ids at 0, attention_mask at 1, labels at 2
        @param steps_per_epoch: number of steps per epoch
        @param n_epochs: number of epochs
        @return: the protocol
        """
        total_steps = n_epochs * steps_per_epoch
        self.scheduler = self.schedulerFactory(total_steps)
        logger = Logger()
        for i in range(n_epochs):
            logger.log(f'epoch {i}')
            self.train(dataloader, logger)
        return logger.getProtocol()


# %%

def getDevice(cuda_desired: bool) -> torch.device:
    """
    @param cuda_desired: True if cuda desired
    @return: cuda if desired and available, cpu otherwise
    """
    return torch.device('cuda') if cuda_desired and torch.cuda.is_available() \
        else torch.device('cpu')


# %%

def readSentencesLabels(filename: str,
                        n_sentences: int,
                        col_sentence: int,
                        col_label: int,
                        delimiter: str = '\t') -> tuple:
    """
    @param filename: file to be read from
    @param delimiter: a delimiter
    @param col_sentence: index of column of sentences
    @param col_label: index of column of labels
    @return: a tuple containing a list of sentences and a list of labels
    """
    df = pd.read_csv(filename, delimiter=delimiter, nrows=n_sentences, header=None)
    return df[col_sentence].values.tolist(), df[col_label].values.tolist()


# %%

def tokenize(sentences: list,
             tokenizer: BertTokenizer,
             max_length: int) -> tuple:
    """
    @param sentences: list of sentences
    @param tokenizer: a tokenizer
    @param max_length: sentences to be padded to
    @return: list of token_ids, list of attention_masks
    Encoding proceeds as follows:
    (1) Tokenize the sentence.
    (2) Prepend the `[CLS]` token to the start.
    (3) Append the `[SEP]` token to the end.
    (4) Map tokens to their IDs.
    (5) Pad or truncate the sentence to `max_length`
    (6) Create attention mask for [PAD] tokens.

    Note: pad_to_max_length is deprecated, no way to get around.
    """
    token_ids = []
    attention_masks = []

    for s in sentences:
        encoded_dict = tokenizer(s,
                                 add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                 truncation=True,
                                 max_length=max_length,
                                 pad_to_max_length=True,
                                 return_attention_mask=True
                                 )
        token_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        # third entry (token_type_ids) not needed

    return token_ids, attention_masks


# %%

def getDataloader(token_ids: list,
                  attention_masks: list,
                  labels: list,
                  split_factor: float,
                  batch_size: int,
                  device: torch.device) -> tuple:
    """
    @param token_ids: token_ids, plain Python list, (n_sentences x max_length)
    @param attention_masks: attention_masks, plain Python list (n_sentences x max_length)
    @param labels: labels, plain Python list (len = n_sentences)
    @param split_factor: share of training sentences
    @param batch_size: size of minibatch
    @param device: device the dataloaders are on
    @return: tuple of two dataloaders, one for training and one for test

    Dataloaders return on each call a list of k 3-tupels (token_ids, attention_mask, label);
    with k = batch_size. All returned elements are torch.tensors on the requested device
    """

    token_ids = torch.tensor(token_ids, device=device)
    attention_masks = torch.tensor(attention_masks, device=device)
    labels = torch.tensor(labels, device=device)

    dataset = TensorDataset(token_ids, attention_masks, labels)
    train_size = int(split_factor * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )

    return train_dataloader, test_dataloader


# %%

def getModule(device: torch.device) -> torch.nn.Module:
    module = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT module, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # do not return attentions weights.
        output_hidden_states=False,  # do not return hidden-states.
    )
    if device == torch.device('cpu'):
        module.cpu()
    else:
        module.cuda()
    return module


# %%

def getTokenizer() -> BertTokenizer:
    """
    @return: the Bert Tokenizer
    """
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# %%

def getOptimizer(module: torch.nn.Module,
                 lr: float,
                 eps: float) -> torch.optim.Optimizer:
    """
    @param module: a module
    @param lr: learning rate
    @param eps: stop criterion
    @return: the Adam optimizer (any other choice is valid)
    """
    return AdamW(module.parameters(), lr=lr, eps=eps)


# %%

def getSchedulerFactory(optimizer: torch.optim.Optimizer) -> Callable:
    """
    @param optimizer: an optimizer
    @return: a function which returns a scheduler depending on the total number of optimizer steps.
    """

    def factory(total_steps: int):
        return get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=total_steps)

    return factory


# %%

# put constants in a dictionary
cfg = {'seed'        : 2,
       'batch_size'  : 32,
       'n_sentences' : 1000,  # number of sentences to read
       'max_length'  : 64,  # max length of sentence (guess or find out)
       'split_factor': 0.8,  # share of training sentences
       'cuda_desired': True,  # True if cuda desired
       'lr'          : 3e-5,  # learning rate of optimizer
       'eps'         : 1e-8,  # stop criterion of optimizer
       'n_epochs'    : None}  # number of epochs

seed = cfg['seed']
batch_size = cfg['batch_size']
n_sentences = cfg['n_sentences']
max_length = cfg['max_length']
split_factor = cfg['split_factor']
cuda_desired = cfg['cuda_desired']
lr = cfg['lr']
eps = cfg['eps']

# seed randomizers
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# %%

# extract at most n sentences and labels; n = n_sentences
unzipped_file = './cola_public/raw/in_domain_train.tsv'
col_sentence = 3  # index of column of sentences
col_label = 1  # index of column of labels

sentences, labels = readSentencesLabels(unzipped_file, n_sentences, col_sentence, col_label)

n = len(sentences)
k = len(list(filter(lambda x: x == 1, labels)))

print(n, k)

# %%

# define the algorithm
device = getDevice(cuda_desired)  # device depending on choice and availability of cuda
module = getModule(device)  # a BERT model or any other
tokenizer = getTokenizer()
optimizer = getOptimizer(module, lr, eps)
schedulerFactory = getSchedulerFactory(optimizer)  # learner can be called with different values of n_epoch
steps_per_epoch = int(len(sentences) * split_factor / batch_size) + 1

# tokenize sentences to token ids and attention masks
token_ids, attention_masks = tokenize(sentences, tokenizer, max_length)

# put token ids, attention masks and labels into a dataloader
train_dataloader, test_dataloader = \
    getDataloader(token_ids, attention_masks, labels, split_factor, batch_size, device)

# %%

# build a learner and get going
cfg['n_epochs'] = 4
learner = Learner(module, optimizer, schedulerFactory, device)
protocol = learner.fit(train_dataloader, steps_per_epoch, cfg['n_epochs'])
print('\nfitting finished')

# make predictions on train and test data
train_labels, train_predictions = learner.predict(train_dataloader)
test_labels, test_predictions = learner.predict(test_dataloader)
log_object = (cfg,
              protocol,
              train_labels, train_predictions,
              test_labels, test_predictions)

# save protocol
log_file = 'log_000.pickle'
with open(log_file, 'wb') as log:
    pickle.dump(log_object, log)

# %%

# How did we do?

log_file = 'log_000.pickle'
with open(log_file, 'rb') as log:
    log_object = pickle.load(log)

cfg, protocol, \
train_labels, train_predictions, \
test_labels, test_predictions = log_object

# show the outcome
print(f"\ntotal number of labels:         {len(train_labels)}\n"
      f"total number of correct labels: {len(list(filter(lambda x: x == 1, train_labels)))}")

starttime = protocol[0][0]
print(f'\n\ntimestamp\tinfo\n')
for timestamp, info in protocol[:]:
    print(f'{timestamp - starttime:.6f}\t{info}')

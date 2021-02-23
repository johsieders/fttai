# Testing torchtext and sentencepiece

import torch
from torch import *
from sentencepiece import *
from torchtext.data import Dataset, Field, Example

from torchtext import data


if __name__ == '__main__':


    # pos = data.TabularDataset(
    # path='data/pos/pos_wsj_train.tsv', format='tsv',
    # fields=[('text', data.Field()),
    #         ('labels', data.Field())])
    #
    # sentiment = data.TabularDataset(
    #     path='data/sentiment/train.json', format='json',
    #     fields={'sentence_tokenized': ('text', data.Field(sequential=True)),
    #              'sentiment_gold': ('labels', data.Field(sequential=False))})

    dict = {"foods": {
        "fruits"    : ["Apple", "Banana"],
        "vegetables": [{"name": "lettuce"}, {"name": "marrow"}]
        }
    }

    fields = {'foods.vegetables.name': ('vegs', data.Field())}
    # dataset = data.TabularDataset(path='sample.json', format='json', fields=fields)
    # v = dataset.examples[0].vegs

    print(fields)

    print(torch.cuda.is_available())
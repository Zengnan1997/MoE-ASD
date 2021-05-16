import pandas as pd
import torch.utils.data as Data
import numpy as np
import torch
BATCH_SIZE = 64
DIMENSION = 300


class TrainLoader:
    def __init__(self, dimension=DIMENSION, batch_size=BATCH_SIZE, tag=False):
        self.dimension = dimension
        self.batch_size = batch_size
        self.tag = tag

    def getTrainLoader(self, wordpairs, data):
        wordpairs = self.getWordsvector(wordpairs, data)
        wordsvector, label = self.DataFrame_to_numpy(wordpairs)
        loader = self.trainloader(wordsvector, label)

        return loader

    def getWordsvectorAndLabel(self, wordpairs, data):
        wordpairs = self.getWordsvector(wordpairs, data)
        print('length of wordpairs is {}'.format(len(wordpairs)))
        wordsvector, label = self.DataFrame_to_numpy(wordpairs)

        return wordsvector, label

    def getWordsvector(self, wordpairs, data):
        wordsvector = pd.DataFrame(columns=wordpairs.columns)
        not_in_set = set()
        not_in_set_index = []
        for index, row in wordpairs.iterrows():
            if row['word1'] in data:
                row['word1'] = data[row['word1']]
            else:
                not_in_set.add(row['word1'])
                not_in_set_index.insert(0, index)
                word1_emb = np.random.rand(self.dimension)
                data[row['word1']] = word1_emb
                row['word1'] = list(word1_emb)

            if row['word2'] in data:
                row['word2'] = data[row['word2']]
            else:
                not_in_set.add(row['word2'])
                not_in_set_index.insert(0, index)
                word2_emb = np.random.rand(self.dimension)
                data[row['word2']] = word2_emb
                row['word2'] = list(word2_emb)
            wordsvector = wordsvector.append(row)
        print(not_in_set)
        print(len(not_in_set))
        return wordsvector

    def DataFrame_to_numpy(self, wordpairs):
        word1array = np.array(wordpairs['word1'].apply(pd.Series))
        word2array = np.array(wordpairs['word2'].apply(pd.Series))

        if self.tag:
            tag = np.array(wordpairs['tag'].apply(pd.Series))

            wordsvector = np.concatenate([word1array, word2array, tag], axis=1)
        else:
            wordsvector = np.concatenate([word1array, word2array], axis=1)

        label = wordpairs['label']
        label = np.array(label, dtype=int)
        wordsvector = torch.from_numpy(wordsvector)
        label = torch.from_numpy(label)

        return wordsvector, label

    def trainloader(self, wordsvector, label):
        torch_dataset = Data.TensorDataset(wordsvector, label)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

        return loader



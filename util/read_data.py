import pandas as pd
import fasttext
import json


class ReadInitialData:
    @staticmethod
    def readfile(file):
        """
        :param file: str, the file need to read
        :return: wordPairs: pandas.DataFrame, columns = ['word1', 'word2', label]
        """
        wordPairs = pd.read_csv(file, sep='\t', names=['word1', 'word2', 'label'])
        wordPairs.fillna('null', inplace=True)

        return wordPairs

    @staticmethod
    def readtagfile(file):
        """
        :param file: str, the file need to read
        :return: wordPairs: pandas.DataFrame, columns = ['word1', 'word2', tag, label]
        """
        wordPairs = pd.read_csv(file, sep='\t', header=None, names=['word1', 'word2', 'tag', 'label'])
        wordPairs.fillna('null', inplace=True)

        return wordPairs

    @staticmethod
    def load_vectors(fname):
        import io
        fin = io.open(fname, 'r', encoding='utf8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        vocab = []
        for line in fin:
            tokens = line.rstrip().split(' ')
            vocab.append(tokens[0])
            data[tokens[0]] = list(map(float, tokens[1:]))

        return data, vocab

    @staticmethod
    def load_vectors_without_head(fname):
        import io
        fin = io.open(fname, 'r', encoding='utf8', newline='\n', errors='ignore')
        data = {}
        vocab = []
        for line in fin:
            tokens = line.rstrip().split(' ')
            vocab.append(tokens[0])
            data[tokens[0]] = list(map(float, tokens[1:]))

        return data, vocab


    @staticmethod
    def loadmodel(modelfile):
        return fasttext.load_model(modelfile)

    @staticmethod
    def loadvocab(vocabfile):
        with open(vocabfile, 'r') as f:
            vocab = f.read().splitlines()
        return vocab

    @staticmethod
    def loadDict(dictFile):
        with open(dictFile, 'r', encoding='utf8') as f:
            dic = json.load(f)
        return dic

    @staticmethod
    def loadEmbed(file):
        embeddf = pd.read_csv(file, sep=',', header=None)
        embeddf.fillna('null', inplace=True)
        embeddf[0] = embeddf[0].str.lower()
        embeddf = embeddf.set_index(0)
        return embeddf

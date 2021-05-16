import os
import pandas as pd
from util.read_data import ReadInitialData


class Symmetry:
    @staticmethod
    def getsymmetrysamples(wordpairs, label, target):
        """
        :param wordpairs: the part of wordpairs need to get symmetry samples
        :param label: symmetry samples should be with label
                        antonym symmetry samples ———— ant_sym ———— 2
                        synonym symmetry samples ———— syn_sym ———— 3
        :param target: the wordpairs with target label should get symmetry samples
        :return: wordpairs +
        """
        targetPairs = wordpairs.loc[wordpairs['label']==target]
        targetPairs.rename(columns={'word1': 'word2', 'word2': 'word1'}, inplace=True)
        targetPairs['label'] = label
        wordpairs = pd.concat([wordpairs, targetPairs], sort=False)
        wordpairs.reset_index(drop=True, inplace=True)

        return wordpairs

    @staticmethod
    def getsymmetrysampleswithtag(wordpairs, label, target):
        """
        :param wordpairs: the part of wordpairs need to get symmetry samples
        :param label: symmetry samples should be with label
                        antonym symmetry samples ———— ant_sym ———— 2
                        synonym symmetry samples ———— syn_sym ———— 3
        :param target: the wordpairs with target label should get symmetry samples
        :return: wordpairs +
        """
        targetPairs = wordpairs.loc[wordpairs['label'] == target]
        targetPairs.rename(columns={'word1': 'word2', 'word2': 'word1'}, inplace=True)
        targetPairs['label'] = label
        targetPairs['tag'] = wordpairs['tag']
        wordpairs = pd.concat([wordpairs, targetPairs], sort=False)
        wordpairs.reset_index(drop=True, inplace=True)

        return wordpairs


if __name__ == '__main__':
    os.chdir('../')
    file = '../dataset/adjective-pairs.train'
    wordpairs = ReadInitialData.readfile(file)
    wordpairs = Symmetry.getsymmetrysamples(wordpairs, 'ant_sym', 1)

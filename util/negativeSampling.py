import pandas as pd
import random


class negativeSample:
    @staticmethod
    def negativeSampling(wordpairs, vocab, negativelabel, target, times):
        """
        :param wordpairs: pd.DataFrame,
        :param vocab: list, unique words in corpus
        :param target: samples with label target need negative sampling
                        antonym negative samples ———— ant_neg ———— -2
                        synonym negative samples ———— syn_neg ———— -3
        :param times: negative samples frequency
        :param negativelabel: the negative samples should be with negativelabel
        :return: wordpairs + negative_samples_df
        """
        length = len(vocab)

        targetPairs = wordpairs[wordpairs['label']==target]
        negative_samples_df = pd.DataFrame(columns=['word1', 'word2', 'label'])
        for index, row in targetPairs.iterrows():
            for i in range(times):
                insert_row = pd.DataFrame([[row['word1'], vocab[random.randint(0, length-1)].strip('\n'), negativelabel]],
                                          columns=['word1', 'word2', 'label'])
                negative_samples_df = negative_samples_df.append(insert_row, ignore_index=True)
            for i in range(times):
                insert_row = pd.DataFrame([[vocab[random.randint(0, length-1)].strip('\n'), row['word2'], negativelabel]],
                                          columns=['word1', 'word2', 'label'])
                negative_samples_df = negative_samples_df.append([insert_row], ignore_index=True)
        wordpairs = pd.concat([wordpairs, negative_samples_df])
        wordpairs.fillna(0, inplace=True)
        wordpairs.reset_index(drop=True, inplace=True)
        return wordpairs



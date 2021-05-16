import argparse
import os
# os.chdir('../')
# import sys
# path = os.getcwd()
# print(path)
# sys.path.append(path)
from util.read_data import ReadInitialData
from util.symmetry_sample import Symmetry
from util.trainloader import TrainLoader
from model.train import Train
import pandas as pd
from model.MoE_model import ConcatModel


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pos', type=str, default='noun')
    parser.add_argument('--embed', type=str, default='fasttext')
    parser.add_argument('--expert_size', type=int, default=96)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--projection_size', type=int, default=4)
    parser.add_argument('--layer_hidden_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    path = './dataset_tag/'

    if args.pos == 'adj':
        train_file = path + 'tag-adjective-pairs.train'
        validation_file = path + 'tag-adjective-pairs.val'
        test_file = path + 'tag-adjective-pairs.test'
    elif args.pos == 'verb':
        train_file = path + 'tag-verb-pairs.train'
        validation_file = path + 'tag-verb-pairs.val'
        test_file = path + 'tag-verb-pairs.test'
    else:
        train_file = path + 'tag-noun-pairs.train'
        validation_file = path + 'tag-noun-pairs.val'
        test_file = path + 'tag-noun-pairs.test'

    print(train_file)
    print(os.getcwd())
    if args.embed == 'fasttext':
        data, vocab = ReadInitialData.load_vectors('./embedding/wiki-news-300d-1M-simple.vec')
    elif args.embed == 'word2vec':
        data, vocab = ReadInitialData.load_vectors('./embedding/GoogleNews-vectors-negative300-simple.vec')
    elif args.embed == 'glove':
        data, vocab = ReadInitialData.load_vectors('./embedding/glove.42B.300d.simple.txt')
    elif args.embed == 'dLCE':
        # tips: the embed size of dLCE is 100
        modelfile = './embedding/wiki_en_dLCE_100d_minFreq_100_simple.bin'
        data, vocab = ReadInitialData.load_vectors(modelfile)
    else:
        print('embed type ERROR')
        raise ValueError

    train_wordpairs = ReadInitialData.readtagfile(train_file)
    validation_wordpairs = ReadInitialData.readtagfile(validation_file)
    test_wordpairs = ReadInitialData.readtagfile(test_file)

    train_sym_wordpairs = Symmetry.getsymmetrysamples(train_wordpairs, label=1, target=1)
    train_sym_wordpairs = Symmetry.getsymmetrysamples(train_sym_wordpairs, label=0, target=0)

    trainLoader = TrainLoader(dimension=args.embed_size, batch_size=args.batch_size, tag=True)

    train_loader = trainLoader.getTrainLoader(train_sym_wordpairs, data)

    train_wordsvector, train_label = trainLoader.getWordsvectorAndLabel(train_wordpairs, data)
    validation_wordsvector, validation_label = trainLoader.getWordsvectorAndLabel(validation_wordpairs, data)
    test_wordsvector, test_label = trainLoader.getWordsvectorAndLabel(test_wordpairs, data)

    Times = 10

    result = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'F1-score'])

    train_model = Train(dimension=args.embed_size, lr=args.lr, decay=args.decay, patience=args.patience)

    for time in range(Times):
        model = ConcatModel(expert_size=args.expert_size, embedding_size=args.embed_size, projection_size=args.projection_size, layer_hidden_size=args.layer_hidden_size)
        model = train_model.train(model, train_loader, train_wordsvector, train_label, validation_wordsvector, validation_label, pos=args.pos)
        result.loc[time] = train_model.test(model, test_wordsvector, test_label)
        print(result.loc[time])
    result.loc[Times] = result.mean()
    print(result)
    result_file = './reuslt_' + args.pos + '.csv'
    result.to_csv(result_file)


if __name__ == "__main__":
    main()


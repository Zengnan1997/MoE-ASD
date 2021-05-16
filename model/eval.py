import torch
import argparse
import gensim
import os
os.chdir('../')
import sys
path = os.getcwd()
# print(path)
sys.path.append(path)
from util.read_data import ReadInitialData
from util.trainloader import TrainLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.MoE_model import ConcatModel


def eval(model, wordsvector, label, dimension):
    word1_vector, word2_vector, tag = wordsvector.split(dimension, 1)
    if torch.cuda.is_available():
        word1_vector, word2_vector, tag = word1_vector.cuda().float(), word2_vector.cuda().float(), tag.cuda().float()
        pred_vector = model(word1_vector, word2_vector, tag)

        pred_vector = torch.sigmoid(pred_vector)
        predict_label = torch.where(pred_vector > 0.5, torch.ones_like(pred_vector), torch.zeros_like(pred_vector))
        predict_label = predict_label.cpu()
    else:
        pred_vector = model(word1_vector.float(), word2_vector.float(), tag.float())
        pred_vector = torch.sigmoid(pred_vector)
        predict_label = torch.where(pred_vector > 0.5, torch.ones_like(pred_vector), torch.zeros_like(pred_vector))

    accuracy = accuracy_score(label, predict_label)
    precision = precision_score(label, predict_label)
    recall = recall_score(label, predict_label)
    f1score = f1_score(label, predict_label)

    return [accuracy, precision, recall, f1score]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_file', type=str, default='./model_noun.pkl')
    parser.add_argument('--pos', type=str, default='noun')
    parser.add_argument('--embed', type=str, default='fasttext')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--expert_size', type=int, default=256)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--projection_size', type=int, default=4)
    parser.add_argument('--layer_hidden_size', type=int, default=20)
    args = parser.parse_args()

    if args.pos == 'adj':
        test_file = './dataset_tag/tag-adjective-pairs.test'
    elif args.pos == 'noun':
        test_file = './dataset_tag/tag-noun-pairs.test'
    elif args.pos == 'verb':
        test_file = './dataset_tag/tag-verb-pairs.test'
    else:
        print('ERROR: Not Exist Pos')
        raise ValueError

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

    model = ConcatModel(expert_size=args.expert_size, embedding_size=args.embed_size, projection_size=args.projection_size, layer_hidden_size=args.layer_hidden_size)
    model.load_state_dict(torch.load(args.model_file))
    trainLoader = TrainLoader(dimension=args.embed_size, batch_size=args.batch_size, tag=True)
    test_wordpairs = ReadInitialData.readtagfile(test_file)
    test_wordsvector, test_label = trainLoader.getWordsvectorAndLabel(test_wordpairs, data)

    model.cuda()
    result = eval(model, test_wordsvector, test_label, dimension=args.embed_size)
    print(result)


if __name__ == '__main__':
    main()
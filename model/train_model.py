import argparse
import os
os.chdir('../')
import sys
path = os.getcwd()
print(path)
sys.path.append(path)
from util.read_data import ReadInitialData
import os
from util.symmetry_sample import Symmetry
from util.trainloader import TrainLoader
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.MoE_model import ConcatModel
import gensim


class Train:
    def __init__(self, dimension, lr, decay, patience):
        self.dimension = dimension
        self.lr = lr
        self.decay = decay
        self.patience = patience

    def getLoss(self, model, wordsvector, label):
        """
        :param wordsvector: torch.tensor
        :param label: torch.tensor
        :return:
        """
        word1_vector, word2_vector, tag = wordsvector.split(self.dimension, 1)
        if torch.cuda.is_available():
            word1_vector, word2_vector, label, tag = word1_vector.cuda().float(), word2_vector.cuda().float(), label.cuda().float(), tag.cuda().float()
            pred_vector = model(word1_vector, word2_vector, tag)
        else:
            pred_vector = model(word1_vector.float(), word2_vector.float(), tag.float())

        return model.loss(pred_vector, label).cpu()

    def train(self, model, train_loader, train_wordsvector, train_label, validation_wordsvector,
              validation_label, pos):
        gpus = [0]
        cuda_gpu = torch.cuda.is_available()
        if cuda_gpu:
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        train_loss = []
        validation_loss = []
        epoch_train_F1 = []
        epoch_validation_F1 = []

        max_validation_f1_score = 0
        last_model = model
        the_last_validation_f1score = 0
        trigger_times = 0

        Iterations = 100
        for epoch in range(Iterations):
            for step, (batch_x, batch_label) in enumerate(train_loader):
                batch_word1, batch_word2, batch_tag = batch_x.split(self.dimension, 1)
                if cuda_gpu:
                    batch_word1, batch_word2, batch_label, batch_tag = batch_word1.cuda(), batch_word2.cuda(), batch_label.cuda().float(), batch_tag.cuda()
                    pred_vector = model(batch_word1.float(), batch_word2.float(), batch_tag.float())
                else:
                    pred_vector = model(batch_word1.float(), batch_word2.float(), batch_tag.float())

                if isinstance(model, torch.nn.DataParallel):
                    model = model.module

                loss = model.loss(pred_vector, batch_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if cuda_gpu:
                torch.cuda.empty_cache()
            if epoch % 15 == 0 and epoch != 0:
                lr = self.lr * (self.decay ** (epoch // 15))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            train_quota = self.test(model, train_wordsvector, train_label)
            epoch_train_F1.append(train_quota[-1])
            epoch_train_loss = self.getLoss(model, train_wordsvector, train_label).item()
            print('epoch {}/{} {} train loss is {}, F1-score is {}'.format(epoch + 1, Iterations, pos, epoch_train_loss, train_quota[-1]))
            train_loss.append(epoch_train_loss)

            validation_quota = self.test(model, validation_wordsvector, validation_label)
            epoch_validation_F1.append(validation_quota[-1])
            epoch_validation_loss = self.getLoss(model, validation_wordsvector, validation_label).item()
            print('epoch {}/{} {} validation loss is {}, F1-score is {}'.format(epoch + 1, Iterations, pos,
                                                                                epoch_validation_loss,
                                                                                validation_quota[-1]))
            validation_loss.append(epoch_validation_loss)

            if validation_quota[-1] > max_validation_f1_score:
                max_validation_f1_score = validation_quota[-1]
                import copy
                last_model = copy.deepcopy(model)

            if validation_quota[-1] <= the_last_validation_f1score:
                trigger_times += 1
                print('trigger times: {}'.format(trigger_times))

                if trigger_times >= self.patience:
                    print('model early stopping.')
                    return last_model
            else:
                print('trigger times: 0')
                trigger_times = 0
            the_last_validation_f1score = validation_quota[-1]

        return last_model

    def test(self, model, wordsvector, label):
        word1_vector, word2_vector, tag = wordsvector.split(self.dimension, 1)
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
    parser.add_argument('--pos', type=str, default='noun')
    parser.add_argument('--embed', type=str, default='fasttext')
    parser.add_argument('--expert_size', type=int, default=256)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--projection_size', type=int, default=4)
    parser.add_argument('--layer_hidden_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--drop_out', type=float, default=0)
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
        model = ConcatModel(expert_size=args.expert_size, embedding_size=args.embed_size, projection_size=args.projection_size, layer_hidden_size=args.layer_hidden_size, drop_out=args.drop_out)
        model = train_model.train(model, train_loader, train_wordsvector, train_label, validation_wordsvector, validation_label, pos=args.pos)
        result.loc[time] = train_model.test(model, test_wordsvector, test_label)
        print(result.loc[time])
    result.loc[Times] = result.mean()
    print(result)
    result_file = './reuslt_' + args.pos + '.csv'
    result.to_csv(result_file)


if __name__ == '__main__':
    main()
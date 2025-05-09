import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import time
import os
from tqdm import tqdm

np.random.seed(int(time.time()))

class Dataloader:
    def __init__(self, datas, targets, fold=5, shuffle=True):
        self.fold = fold
        self.flag = 0
        self.n_class = len(set(targets))

        if isinstance(datas, list):
            datas = np.array(datas, dtype=float)
        if isinstance(targets, list):
            targets = np.array(targets, dtype=int)

        if shuffle:
            index = np.arange(len(datas))
            np.random.shuffle(index)
            datas = datas[index]
            targets = targets[index]

        fold_size = len(datas) // fold
        self.datas = []
        self.targets = []
        for i in range(fold):
            if i == fold - 1:
                self.datas.append(datas[i * fold_size:])
                self.targets.append(targets[i * fold_size:])
            else:
                self.datas.append(datas[i * fold_size:(i + 1) * fold_size])
                self.targets.append(targets[i * fold_size:(i + 1) * fold_size])

    def get_train_test(self, normalize=True):
        train_data = None
        for i in range(self.fold):
            if i == self.flag:
                test_data = self.datas[i]
                test_target = self.targets[i]
            elif train_data is None:
                train_data = self.datas[i]
                train_target = self.targets[i]
            else:
                train_data = np.concatenate((train_data, self.datas[i]), axis=0)
                train_target = np.concatenate((train_target, self.targets[i]), axis=0)
        self.flag = (self.flag + 1) % self.fold
        train_target = self.one_hot_encode(train_target)
        test_target = self.one_hot_encode(test_target)

        if normalize:
            mean = np.mean(train_data, axis=0)
            std = np.std(train_data, axis=0)
            std[std == 0] = 1e-10
            train_data = (train_data - mean) / std
            test_data = (test_data - mean) / std

        return train_data, train_target, test_data, test_target

    def one_hot_encode(self, targets):
        one_hot = np.zeros((len(targets), self.n_class))
        one_hot[np.arange(len(targets)), targets] = 1
        return one_hot

    def get_input_output_dims(self):
        return self.datas[0].shape[1], self.n_class


class MLP:
    def __init__(self, dataloader: Dataloader, hidden_size: int, regularization=None, lambda_reg=0.01,
                 fold=5, lr=0.001, max_epoch=1000, early=100, gamma=1.0):
        self.input_size, self.output_size = dataloader.get_input_output_dims()
        self.hidden_size = hidden_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.max_epoch = max_epoch
        self.early = early
        self.gamma = gamma
        self.dataloader = dataloader
        self.fold = fold

        self.w1, self.w2, self.b1, self.b2 = None, None, None, None
        self.intermediate = None

    def initialize_weights(self):
        limit1 = np.sqrt(6 / self.input_size)
        self.w1 = np.random.uniform(-limit1, limit1, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))

        limit2 = np.sqrt(6 / self.hidden_size)
        self.w2 = np.random.uniform(-limit2, limit2, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))

        self.intermediate = None

    def relu(self, z):
        return np.maximum(z, 0)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x, no_grad=False):
        z1 = x @ self.w1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self.softmax(z2)
        if not no_grad:
            self.intermediate = (x, z1, a1, z2, a2)
        return a2

    def back_prop(self, y):
        x, z1, a1, z2, a2 = self.intermediate
        N = x.shape[0]
        delta2 = (a2 - y) / N  # (N, output_size)
        dw2 = a1.T @ delta2  # (hidden_size, output_size)
        db2 = np.sum(delta2, axis=0, keepdims=True)  # (1, output_size)
        delta1 = delta2 @ self.w2.T * self.relu_derivative(z1)  # (N, hidden_size)
        dw1 = x.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        if self.regularization == 'L1':
            dw1 += self.lambda_reg * np.sign(self.w1)
            dw2 += self.lambda_reg * np.sign(self.w2)
        elif self.regularization == 'L2':
            dw1 += self.lambda_reg * self.w1
            dw2 += self.lambda_reg * self.w2

        self.w1 -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

    def cross_entropy_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))

    def L1_loss(self):
        return np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))

    def L2_loss(self):
        return np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2)

    def accuracy(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true)

    def train(self):
        start = time.time()
        cross_val_train_acc, cross_val_test_acc = [], []
        print(f"\n----------------------------------------------------------------------\n"
              f"Start training MLP: {self.fold} fold, regularization: {self.regularization}, lr: {self.lr}, "
              f"hidden layers: {self.hidden_size}, early stopping: {self.early}, gamma: {self.gamma} ...")

        for cross_fold_i in range(self.fold):
            self.initialize_weights()
            lr = self.lr

            train_data, train_target, test_data, test_target = self.dataloader.get_train_test()
            train_accuracy_record, train_loss_record = [], []
            test_accuracy_record, test_loss_record = [], []
            L1_loss_record, L2_loss_record = [], []

            best_epoch = 0
            best_test_accuracy = 0
            not_improved = 0

            bar = tqdm(range(self.max_epoch))
            bar.set_description(f"Fold {cross_fold_i + 1}/{self.fold}")
            for epoch in bar:
                y_pred = self.forward(train_data)
                self.back_prop(train_target)
                lr *= self.gamma

                train_loss = self.cross_entropy_loss(y_pred, train_target)
                train_acc = self.accuracy(y_pred, train_target)
                test_pred = self.forward(test_data, no_grad=True)
                test_loss = self.cross_entropy_loss(test_pred, test_target)
                test_acc = self.accuracy(test_pred, test_target)

                train_loss_record.append(train_loss)
                train_accuracy_record.append(train_acc)
                test_loss_record.append(test_loss)
                test_accuracy_record.append(test_acc)

                L1_loss_record.append(self.L1_loss())
                L2_loss_record.append(self.L2_loss())

                if test_acc > best_test_accuracy:
                    best_epoch = epoch
                    best_test_accuracy = test_acc
                    not_improved = 0
                else:
                    not_improved += 1
                    if not_improved > self.early:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                bar.set_postfix(train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}",
                                test_loss=f"{test_loss:.4f}", test_acc=f"{test_acc:.4f}", lr=f"{lr:.4e}")
            bar.close()

            cross_val_train_acc.append(train_accuracy_record[best_epoch])
            cross_val_test_acc.append(best_test_accuracy)

            print(f'Weights are taken from the best epoch:{best_epoch + 1}, '
                  f'test accuracy = {best_test_accuracy:.6f}, train accuracy = {train_accuracy_record[best_epoch]:.6f}')

            plt.figure(figsize=(8, 6))
            plt.plot(train_loss_record, label='train loss')
            plt.plot(test_loss_record, label='test loss')
            plt.axvline(x=best_epoch, color='red', linestyle='--', label='best epoch')
            plt.title(f'Loss at fold {cross_fold_i + 1}')
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            plt.text(xlim[0] + (xlim[1] - xlim[0]) * 0.7, ylim[0] + (ylim[1] - ylim[0]) * 0.3,
                     f'Regularization: {self.regularization} \nLambda: {self.lambda_reg} '
                     f'\nlr: {self.lr} \ngamma: {self.gamma} \nhidden size: {self.hidden_size}')
            plt.legend()
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.plot(train_accuracy_record, label='train accuracy')
            plt.plot(test_accuracy_record, label='test accuracy')
            plt.axvline(x=best_epoch, color='red', linestyle='--', label='best epoch')
            plt.title(f'Accuracy at fold {cross_fold_i + 1}')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            plt.text(xlim[0] + (xlim[1] - xlim[0]) * 0.7, ylim[0] + (ylim[1] - ylim[0]) * 0.3,
                     f'Regularization: {self.regularization} \nLambda: {self.lambda_reg} \nlr: {self.lr} \ngamma: '
                     f'{self.gamma} \nhidden size: {self.hidden_size} \nbest test accuracy: {best_test_accuracy:.4f}')
            plt.legend()
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.plot(L1_loss_record, label='L1 loss')
            plt.plot(L2_loss_record, label='L2 loss')
            plt.axvline(x=best_epoch, color='red', linestyle='--', label='best epoch')
            plt.title(f'Regularization at fold {cross_fold_i + 1}')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            plt.text(xlim[0] + (xlim[1] - xlim[0]) * 0.7, ylim[0] + (ylim[1] - ylim[0]) * 0.3,
                     f'Regularization: {self.regularization} \nLambda: {self.lambda_reg} \nlr: {self.lr} \ngamma: '
                     f'{self.gamma} \nhidden size: {self.hidden_size}')
            plt.legend()
            plt.show()

        duration = time.time() - start
        print(f'Cross-validation results:\n'
              f'test accuracy: {np.mean(cross_val_test_acc):.4f} ± {np.std(cross_val_test_acc):.4f}, '
              f'train accuracy: {np.mean(cross_val_train_acc):.4f} ± {np.std(cross_val_train_acc):.4f}')
        print(f'Training time: {duration // 60:.0f} mins, {duration % 60:.0f} seconds.')


def main():
    with open(os.path.join('LFW dataset', 'lfw.pkl'), 'rb') as file:
        flw_people = pickle.load(file)  # ['data', 'images', 'target', 'target_names', 'DESCR']

    dataloader = Dataloader(flw_people['data'], flw_people['target'], fold=5)
    mlp = MLP(dataloader, hidden_size=256, regularization=None,
              fold=5, lr=0.1, max_epoch=5000, early=300, gamma=0.99)
    mlp.train()

    mlp = MLP(dataloader, hidden_size=256, regularization='L1', lambda_reg=0.01,
              fold=5, lr=0.1, max_epoch=3000, early=300, gamma=0.99)
    mlp.train()

    mlp = MLP(dataloader, hidden_size=256, regularization='L2', lambda_reg=0.01,
              fold=5, lr=0.1, max_epoch=3000, early=300, gamma=0.99)
    mlp.train()

    mlp = MLP(dataloader, hidden_size=512, regularization='L2', lambda_reg=0.01,
              fold=5, lr=0.1, max_epoch=3000, early=300, gamma=0.99)
    mlp.train()

    mlp = MLP(dataloader, hidden_size=1024, regularization='L2', lambda_reg=0.01,
              fold=5, lr=0.1, max_epoch=3000, early=300, gamma=0.99)
    mlp.train()



if __name__ == '__main__':
    main()

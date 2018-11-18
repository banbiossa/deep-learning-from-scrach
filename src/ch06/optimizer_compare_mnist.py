# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from ..dataset.mnist import load_mnist
from ..common.util import smooth_curve
from ..common.multi_layer_net import MultiLayerNet
from ..common.optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSprop
from ..common.optimizer import Adam
import logging

logger = logging.getLogger(__name__)


def main():
        logger = logging.getLogger(__name__)

        # 0:MNISTデータの読み込み==========
        logger.info("0:MNISTデータの読み込み==========")
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

        train_size = x_train.shape[0]
        batch_size = 128
        max_iterations = 2000

        # 1:実験の設定==========
        logger.info("1:実験の設定==========")
        optimizers = {}
        optimizers['SGD'] = SGD()
        optimizers['Momentum'] = Momentum()
        optimizers['AdaGrad'] = AdaGrad()
        optimizers['Adam'] = Adam()
        #optimizers['RMSprop'] = RMSprop()
        logger.info(f"Optimizers: {optimizers}")

        networks = {}
        train_loss = {}
        for key in optimizers.keys():
                networks[key] = MultiLayerNet(
                    input_size=784, hidden_size_list=[100, 100, 100, 100],
                    output_size=10)
                train_loss[key] = []

        # 2:訓練の開始==========
        logger.info(f"2:訓練の開始===={max_iterations} iters ==== ")
        for i in range(max_iterations):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]

                for key in optimizers.keys():
                        grads = networks[key].gradient(x_batch, t_batch)
                        optimizers[key].update(networks[key].params, grads)

                        loss = networks[key].loss(x_batch, t_batch)
                        train_loss[key].append(loss)

                if i % 100 == 0:
                        logger.info(f"==== iteration: {i} =====")
                        for key in optimizers.keys():
                                loss = networks[key].loss(x_batch, t_batch)
                                print(key + ":" + str(loss))

        # 3.グラフの描画==========
        logger.info("3.グラフの描画==========")
        markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
        x = np.arange(max_iterations)
        for key in optimizers.keys():
                plt.plot(x, smooth_curve(train_loss[key]),
                         marker=markers[key], markevery=100, label=key)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()


if __name__ == "__main__":
        logger.info("Kick main")
        main()

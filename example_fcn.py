import argparse
import sys

import numpy as np

from raylib import rl

from tinygrad import Tensor, TinyJit
from tinygrad.helpers import trange
from tinygrad import nn
from tinygrad.nn import Linear, optim, state
from tinygrad.nn.datasets import mnist

from gym import render, init_render, close_render

# reproduce weights and biases
Tensor.manual_seed(42)

# fully-connected neural network
class FCNN:
    def __init__(self, arch: list, bias: bool):
        self.arch = arch
        self.bias = bias
        self.nn = []
        for k in range(len(arch)-1):
            self.nn.append(
                Linear(self.arch[k], self.arch[k+1], bias=self.bias),
            )
    def __call__(self, x):
        for layer in self.nn:
            x = layer(x)
        x = x.sigmoid()
        return x

    def show_dims(self):
        for k in range(len(self.nn)):
            if hasattr(self.nn[k], "weight"):
                print("weight:", self.nn[k].weight.numpy().shape)
            if hasattr(self.nn[k], "bias"):
                print("bias:", self.nn[k].bias.numpy().shape)

def train(model, X_train, Y_train, optim, batch_size, steps, loss_fn):
  
    @TinyJit
    @Tensor.train()
    def train_step(X_train, Y_train) -> Tensor:
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        x, y = X_train[samples], Y_train[samples]
        out = model.forward(x) if hasattr(model, "forward") else model(x)

        optim.zero_grad()
        loss = loss_fn(out, y).backward()
        optim.step()
        return loss

    @TinyJit
    @Tensor.test()
    def train_acc(X_train, Y_train) -> Tensor:
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        x, y = X_train[samples], Y_train[samples]
        acc = ((model(x).sub(y)).abs() <= 0.5).mean()*100
        return acc

    for k in range(steps):
        loss = train_step(X_train, Y_train)
        acc = train_acc(X_train, Y_train)
        # render nn
        render(model, k+1, loss.item(), acc.item())

    print(model(X_train).numpy(), Y_train.numpy())
    return None

def main(opts: argparse.Namespace) -> None:
    if opts.debug:
        print("CLI options:", opts)
        from tinygrad import Device
        print("device:", Device.DEFAULT)

    # load dataset
    # X_train, Y_train, X_test, Y_test = mnist()
    # X_train = X_train.reshape((-1, 28 * 28))
    # X_train, Y_train = X_train[:1000,:], Y_train[:1000]

    # XOR gate
    X_train = Tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    Y_train = Tensor([
        [0],
        [1],
        [1],
        [0]
    ])

    # neural network of fully-connected layers
    net = FCNN(arch=opts.arch, bias=True)
    net.show_dims()

    # optimizer
    optimizer = optim.SGD(state.get_parameters(net), lr=0.001)
    
    # window init
    init_render()

    # keep window open during training
    # while not rl.WindowShouldClose():
    # make training step
    train(net, X_train, Y_train, optimizer, opts.batch_size, opts.steps, lambda out,y: out.sparse_categorical_crossentropy(y))
    # break

    input()
    close_render()

if __name__ == "__main__":
    # command line arguments
    args = argparse.ArgumentParser()
    args.add_argument("--arch", nargs='+', type=int, help="shape of FC neural network")
    args.add_argument("--batch-size", type=int, help="batch size for training stage")
    args.add_argument("--steps", type=int, help="training steps")
    args.add_argument("--debug", "-d", action="store_true", default=False, help="debugging flag")
    opts = args.parse_args()

    main(opts)

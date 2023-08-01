import abc
import numpy as np


class Optimizer(abc.ABC):

    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass


class SGD(Optimizer):

    def __init__(self, parameters, lr=0.001):
        super(self.__class__, self).__init__(parameters, lr=lr)

    def step(self):
        for _, param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for _, param in self.parameters:
            if param.requires_grad:
                param.grad.fill(0)


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super(self.__class__, self).__init__(parameters, lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        for index, (name, param) in enumerate(self.parameters):
            if param.requires_grad:
                ext_name = f'{index}.{name}'
                if ext_name not in self.m:
                    self.m[ext_name] = np.zeros_like(param.data)
                    self.v[ext_name] = np.zeros_like(param.data)

                # Update biased first and second moments estimates
                self.m[ext_name] = self.beta1 * self.m[ext_name] + (1 - self.beta1) * param.grad
                self.v[ext_name] = self.beta2 * self.v[ext_name] + (1 - self.beta2) * (param.grad ** 2)

                # Bias correction
                m_hat = self.m[ext_name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[ext_name] / (1 - self.beta2 ** self.t)

                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for _, param in self.parameters:
            if param.requires_grad:
                param.grad.fill(0)

import abc
from collections import OrderedDict
import numpy as np
import stensor
import stensor.grad_functions as GF


class Module(abc.ABC):

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules    = OrderedDict()

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = '  '.join(repr(module).splitlines(True))
            child_lines.append(f'({key}): ' + mod_str)
        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def extra_repr(self):
        return ''

    def __setattr__(self, name, value):
        if issubclass(type(value), Module):
            self.add_module(name, value)
        super().__setattr__(name, value)

    def add_module(self, name, module):
        self._modules[name] = module

    def parameters(self):
        params = list(self._parameters.items())
        for name in self._modules.keys():
            params += self._modules[name].parameters()
        return params

    def __call__(self, *input):
        return self.forward(*input)

    @abc.abstractmethod
    def forward(self, *input):
        pass


class Linear(Module):

    def __init__(self, in_features, out_features):
        super(self.__class__, self).__init__()
        self.in_features, self.out_features = in_features, out_features

        self._parameters['weight'] = stensor.create(np.random.randn(in_features, out_features), requires_grad=True)
        self._parameters['bias']   = stensor.create(np.random.randn(out_features), requires_grad=True)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

    def forward(self, x):
        weight, bias = self._parameters['weight'], self._parameters['bias']
        return x @ weight + bias


class Sigmoid(Module):

    def forward(self, x):
        return GF.SigmoidFn().forward(x)


class MSELoss(Module):

    def forward(self, input, target):
        return GF.MSELossFn().forward(input, target)

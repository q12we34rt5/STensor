import numpy as np
import stensor.grad_functions as GF


class Tensor:

    def __init__(self, data, requires_grad=False, is_leaf=True, grad_fn=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None

        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn

        if self.requires_grad:
            if self.is_leaf:
                self.grad = np.zeros_like(self.data, dtype=np.float32)
        else:
            self.grad_fn = None

    def __repr__(self):
        padding = ' ' * (len(self.__class__.__name__))
        main_str = self.__class__.__name__ + '('
        lines = str(self.data).split('\n')
        if len(lines) == 1:
            main_str += lines[0]
        else:
            main_str += f'\n{padding} '.join(lines)
        extra_repr = ''
        if type(self.grad_fn) != type(None):
            extra_repr += f', grad_fn={self.grad_fn}'
        elif self.requires_grad and self.is_leaf:
                extra_repr += f', requires_grad={self.requires_grad}'
        if extra_repr:
            main_str += extra_repr
        main_str += ')'
        return main_str

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if type(grad) == type(None):
            grad = np.ones_like(self.data, dtype=np.float32)
        if self.is_leaf:
            self.grad += grad
        if type(self.grad_fn) != type(None):
            self.grad_fn.backward(grad)

    @staticmethod
    def _to_stensor(v, **kwargs):
        if type(v) != Tensor:
            return create(v, **kwargs)
        return v

    def __add__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.AddFn().forward(self, v)

    def __radd__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.AddFn().forward(v, self)

    def __sub__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.SubFn().forward(self, v)

    def __rsub__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.SubFn().forward(v, self)

    def __mul__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.MulFn().forward(self, v)

    def __rmul__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.MulFn().forward(v, self)

    def __truediv__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.DivFn().forward(self, v)

    def __rtruediv__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.DivFn().forward(v, self)

    def __pow__(self, v):
        v = self._to_stensor(v, requires_grad=False)
        return GF.PowFn().forward(self, v)

    def __neg__(self):
        return GF.NegFn().forward(self)

    def exp(self):
        return GF.ExpFn().forward(self)

    def log(self):
        return GF.LogFn().forward(self)


def create_intermediate(data, requires_grad=False, grad_fn=None, **kwargs):
    return Tensor(data, requires_grad=requires_grad, is_leaf=False, grad_fn=grad_fn, **kwargs)


def create(data, requires_grad=False, **kwargs):
    return Tensor(data, requires_grad=requires_grad, is_leaf=True, grad_fn=None, **kwargs)

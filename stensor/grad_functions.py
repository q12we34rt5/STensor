import abc
import numpy as np
import stensor
from stensor.utils import reduce_dim


class GradFunction(abc.ABC):

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    @abc.abstractmethod
    def forward(self, *args):
        pass

    @abc.abstractmethod
    def backward(self, grad):
        pass


class AddFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2
        data = v1.data + v2.data
        self.data_shape = data.shape
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        if self.v1.requires_grad:
            self.v1.backward(reduce_dim(grad, self.v1.data.shape))
        if self.v2.requires_grad:
            self.v2.backward(reduce_dim(grad, self.v2.data.shape))


class SubFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2
        data = v1.data - v2.data
        self.data_shape = data.shape
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        if self.v1.requires_grad:
            self.v1.backward(reduce_dim(grad, self.v1.data.shape))
        if self.v2.requires_grad:
            self.v2.backward(reduce_dim(-grad, self.v2.data.shape))


class MulFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2
        self.v1_data, self.v2_data = v1.data.copy(), v2.data.copy()
        data = v1.data * v2.data
        self.data_shape = data.shape
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        if self.v1.requires_grad:
            self.v1.backward(reduce_dim(grad * self.v2_data, self.v1.data.shape))
        if self.v2.requires_grad:
            self.v2.backward(reduce_dim(grad * self.v1_data, self.v2.data.shape))


class DivFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2
        data = v1.data / v2.data
        self.data_shape = data.shape
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        if self.v1.requires_grad:
            self.v1.backward(reduce_dim(grad / self.v2.data, self.v1.data.shape))
        if self.v2.requires_grad:
            self.v2.backward(reduce_dim(-grad * self.v1.data / (self.v2.data ** 2), self.v2.data.shape))


class PowFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2
        self.v1_data, self.v2_data = v1.data.copy(), v2.data.copy()
        data = v1.data ** v2.data
        self.data_shape = data.shape
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        if self.v1.requires_grad:
            self.v1.backward(reduce_dim(grad * self.v2_data * self.v1_data ** (self.v2_data - 1), self.v1.data.shape))
        if self.v2.requires_grad:
            self.v2.backward(reduce_dim(grad * self.v1_data ** self.v2_data * np.log(self.v1_data), self.v2.data.shape))


class NegFn(GradFunction):

    def forward(self, v):
        self.v = v
        data = -v.data
        self.data_shape = data.shape
        requires_grad = v.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        self.v.backward(-grad)


class MatmulFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2

        self.v1_data = (v1.data[None]   if v1.data.ndim == 1 else v1.data).copy()
        self.v2_data = (v2.data[None].T if v2.data.ndim == 1 else v2.data).copy()

        data = self.v1_data @ self.v2_data

        self.data_shape = data.shape
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        if self.v1.requires_grad:
            v1_grad = reduce_dim(grad @ np.moveaxis(self.v2_data, -1, -2), self.v1_data.shape)
            if self.v1.data.ndim == 1:
                v1_grad = v1_grad[0]
            self.v1.backward(v1_grad)

        if self.v2.requires_grad:
            v2_grad = reduce_dim(np.moveaxis(self.v1_data, -1, -2) @ grad, self.v2_data.shape)
            if self.v2.data.ndim == 1:
                v2_grad = v2_grad.T[0]
            self.v2.backward(v2_grad)


class SumFn(GradFunction):

    def forward(self, v, axis=None, keepdims=False):
        self.v = v
        data = v.data.sum(axis=axis, keepdims=True)
        self.data_shape = data.shape
        if not keepdims:
            data = data.squeeze()
        requires_grad = v.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        # assert grad.shape == self.data_shape
        grad = grad.reshape(self.data_shape)
        self.v.backward(grad * np.ones_like(self.v.data))


class MeanFn(GradFunction):

    def forward(self, v, axis=None, keepdims=False):
        self.v = v
        data = v.data.mean(axis=axis, keepdims=True)
        self.data_shape = data.shape
        if not keepdims:
            data = data.squeeze()
        requires_grad = v.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        import functools
        # assert grad.shape == self.data_shape
        # reduced_size = np.prod(self.v.data.shape) / np.prod(self.data_shape)
        reduced_size = functools.reduce(lambda x, y: x * y, self.v.data.shape) / functools.reduce(lambda x, y: x * y, self.data_shape)
        grad = grad.reshape(self.data_shape)
        self.v.backward(grad * np.full(self.v.data.shape, 1 / reduced_size))


class ExpFn(GradFunction):

    def forward(self, v):
        self.v = v
        data = np.exp(v.data)
        self.data = data # backup
        self.data_shape = data.shape
        requires_grad = v.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        self.v.backward(grad * self.data)


class LogFn(GradFunction):

    def forward(self, v):
        self.v = v
        self.v_data = self.v.data.copy()
        data = np.log(v.data)
        self.data_shape = data.shape
        requires_grad = v.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        self.v.backward(grad / self.v_data)


class SigmoidFn(GradFunction):

    def forward(self, v):
        self.v = v
        data = 1 / (1 + np.exp(-v.data))
        self.data = data # backup
        self.data_shape = data.shape
        requires_grad = v.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == self.data_shape
        self.v.backward(grad * self.data * (1 - self.data))


class MSELossFn(GradFunction):

    def forward(self, v1, v2):
        self.v1, self.v2 = v1, v2
        assert v1.data.shape == v2.data.shape
        self.data = v1.data - v2.data
        self.data_shape = self.data.shape
        data = (self.data ** 2).sum()
        requires_grad = v1.requires_grad | v2.requires_grad
        return stensor.create_intermediate(data, requires_grad=requires_grad, grad_fn=self)

    def backward(self, grad):
        assert grad.shape == ()
        if self.v1.requires_grad or self.v2.requires_grad:
            grad = grad * self.data
            if self.v1.requires_grad:
                self.v1.backward(grad)
            if self.v2.requires_grad:
                self.v2.backward(-grad)

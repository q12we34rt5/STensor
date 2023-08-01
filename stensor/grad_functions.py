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

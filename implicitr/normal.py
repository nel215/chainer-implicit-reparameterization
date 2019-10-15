import chainer
from chainer.distributions import Normal as _Normal
from chainer import functions as F
from chainer import as_variable
from chainer import FunctionNode
from chainer import backend


class SampleNormal(FunctionNode):

    def forward(self, inputs):
        loc, scale = inputs
        xp = backend.get_array_module(loc)
        eps = xp.random.randn(*loc.shape)
        z = as_variable(loc).array + eps * as_variable(scale).array

        self.retain_inputs((0, 1))
        self.retain_outputs((0,))
        return xp.array(z, dtype='f'),

    def backward(self, target_input_indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        loc, scale = inputs
        z, = self.get_retained_outputs()
        d = _Normal(loc, scale)
        with chainer.using_config('enable_backprop', True):
            cdf = d.cdf(z.array)
        dcdf = chainer.grad([cdf], inputs, grad_outputs)
        pdf = d.prob(z)
        for i, dphi in enumerate(dcdf):
            dcdf[i] = -dphi / pdf
        return dcdf


def sample_normal(loc, scale):
    return SampleNormal().apply((loc, scale))[0]


class Normal(_Normal):

    def sample_n(self, n):
        res = []
        for i in range(n):
            res.append(sample_normal(self.loc, self.scale))
        return F.hstack(res)

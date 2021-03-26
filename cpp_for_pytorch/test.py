from torch.autograd import Function
import torch
import test_cpp


class _TestFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        """
        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
        The context can be used to store tensors that can be then retrieved
        during the backward pass.
        """
        return test_cpp.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = test_cpp.backward(gradOutput)
        return gradX, gradY

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()

    def forward(self, inputA, inputB):
        return _TestFunction.apply(inputA, inputB)


# test
test = Test()
a = torch.Tensor([1])
b = torch.Tensor([1])
z = test(a,b)
print(a, b, z)
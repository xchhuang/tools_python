#include "test.h"

// part1:forward propagation
torch::Tensor Test_forward_cpu(const torch::Tensor& x, const torch::Tensor& y)
{
    AT_ASSERTM(x.sizes() == y.sizes());
    torch::Tensor z = torch::zeros(x.sizes());
    z = 2 * x + y;
    return z;
}

//part2:backward propagation
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput)
{
    torch::Tensor gradOutputX = 2 * gradOutput * torch::ones(gradOutput.sizes());
    torch::Tensor gradOutputY = gradOutput * torch::ones(gradOutput.sizes());
    return {gradOutputX, gradOutputY};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &Test_forward_cpu, "Test forward");
    m.def("backward", &Test_backward_cpu, "Test backward");
}
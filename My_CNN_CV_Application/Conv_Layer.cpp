#include "Conv_Layer.h"

Conv_Layer::Conv_Layer()
{
}

Conv_Layer::Conv_Layer(Tensor inputTensor, Tensor outputTensor, Tensor filterTensor) : Layer(inputTensor, outputTensor)
{
}

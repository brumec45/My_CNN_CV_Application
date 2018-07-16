#include "Conv_Layer.h"

Conv_Layer::Conv_Layer()
{
}

Conv_Layer::Conv_Layer(Tensor *inputTensor, Tensor *outputTensor, Tensor *kernelTensor, int stride, int padding) : Layer(inputTensor, outputTensor, stride, padding)
{
	this->kernelTensor = kernelTensor;
}

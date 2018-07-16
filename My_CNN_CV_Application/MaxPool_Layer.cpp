#include "MaxPool_Layer.h"

MaxPool_Layer::MaxPool_Layer()
{
}

MaxPool_Layer::MaxPool_Layer(Tensor * inputTensor, Tensor * outputTensor, int stride, int padding) : Layer(inputTensor, outputTensor, stride, padding)
{
	this->stride = stride;
}

int MaxPool_Layer::GetStride()
{
	return this->stride;
}

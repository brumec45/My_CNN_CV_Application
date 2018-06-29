#include "MaxPool_Layer.h"

MaxPool_Layer::MaxPool_Layer()
{
}

MaxPool_Layer::MaxPool_Layer(Tensor * inputTensor, Tensor * outputTensor, int stride) : Layer(inputTensor, outputTensor)
{
	this->stride = stride;
}

int MaxPool_Layer::GetStride()
{
	return this->stride;
}

#include "Layer.h"

Layer::Layer()
{
}

Layer::Layer(Tensor *inputTensor, Tensor *outputTensor, int stride, int padding)
{
	this->inputTensor = inputTensor;
	this->outputTensor = outputTensor;
	this->stride = stride;
	this->padding = padding;
}


void Layer::SetPreviousLayer(Layer * previousLayer)
{
	this->previousLayer = previousLayer;
}

void Layer::SetNextLayer(Layer * nextLayer)
{
	this->nextLayer = nextLayer;
}

Tensor* Layer::GetInputTensor()
{
	return this->inputTensor;
}

Tensor* Layer::GetOutputTensor()
{
	return this->outputTensor;
}

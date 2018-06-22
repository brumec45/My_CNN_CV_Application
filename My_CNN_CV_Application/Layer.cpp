#include "Layer.h"

Layer::Layer()
{
}

Layer::Layer(Tensor *inputTensor, Tensor *outputTensor)
{
	this->inputTensor = inputTensor;
	this->outputTensor = outputTensor;
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
	return this->outputTensor;
}

Tensor* Layer::GetOutputTensor()
{
	return this->inputTensor;
}

#pragma once
#include "Tensor.h"
class Layer {
protected:
	Layer* previousLayer, *nextLayer;
	Tensor inputTensor, outputTensor;
public:
	Layer();
	Layer(Tensor inputTensor, Tensor outputTensor);
	void SetPreviousLayer(Layer* previousLayer);
	void SetNextLayer(Layer* nextLayer);
	Tensor GetInputTensor();
	Tensor GetOutputTensor();
};
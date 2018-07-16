#pragma once
#include "Layer.h"

class MaxPool_Layer : public Layer {
protected:
	
public:
	MaxPool_Layer();
	MaxPool_Layer(Tensor *inputTensor, Tensor *outputTensor, int stride, int padding);
	
	int GetStride();
};
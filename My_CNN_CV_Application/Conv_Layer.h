#pragma once
#include "Layer.h"
class Conv_Layer: public Layer {
protected:
	Tensor *kernelTensor;
public:
	Conv_Layer();
	Conv_Layer(Tensor *inputTensor, Tensor *outputTensor, Tensor *kernelTensor, int stride, int padding);
};
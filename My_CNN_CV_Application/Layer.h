#pragma once
#include "Tensor.h"
class Layer {
protected:
	Layer* previousLayer, *nextLayer;
	Tensor* inputTensor, *outputTensor;
public:
	Layer();
	Layer(Tensor *inputTensor, Tensor *outputTensor);

	virtual float * GetOutputGPUMemoryPointer() = 0;
	virtual void Forward() = 0;
	virtual void SetInputData(float* inputData) = 0;

	void SetPreviousLayer(Layer* previousLayer);
	void SetNextLayer(Layer* nextLayer);

	virtual float* GetLayerOutputData() = 0;
	Tensor* GetInputTensor();
	Tensor* GetOutputTensor();
};
#pragma once
#include "TensorDimension.h"
#include <random>

class Tensor {
private:
	float * tensorData = nullptr;
	int tensorSize;
	TensorDimension tensorDimension;
	
public:
	Tensor();
	Tensor(TensorDimension tensorDimension);
	void InitKernelWeights();
	void SetKernelWeights(float* weights);
	void SetTensorData(float * tensorData);
	void SetTensorDimension(TensorDimension tensorDimension);
	float* GetTensorData();
	int GetBatchDimension();
	int GetChannels();
	int GetHeight();
	int GetWidth();
	int GetTensorSize();
	TensorDimension GetTensorDimension();
};
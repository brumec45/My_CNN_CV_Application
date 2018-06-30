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
	Tensor(const TensorDimension &tensorDimension);
	void InitKernelWeights();
	void SetTensorData(float * tensorData);
	void SetTensorDimension(const TensorDimension &tensorDimension);
	float* GetTensorData();
	int GetBatchDimension();
	int GetChannels();
	int GetHeight();
	int GetWidth();
	int GetTensorSize();
};
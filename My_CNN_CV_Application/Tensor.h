#pragma once
#include "TensorDimension.h"

class Tensor {
private:
	float * tensorData = nullptr;
	TensorDimension tensorDimension;
public:
	Tensor();
	Tensor(int batchDimension, int channels, int height, int width);
	void SetTensorData(float * tensor);
	void SetTensorDimension(TensorDimension tensorDimension);
	float* GetTensorData() const;
	TensorDimension GetTensorDimension();
};
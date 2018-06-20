#include "Tensor.h"

Tensor::Tensor()
{
}

Tensor::Tensor(int batchDimension, int channels, int height, int width)
{
	this->tensorData = new float[batchDimension * channels * height * width];
	this->tensorDimension.batchDimension = batchDimension;
	this->tensorDimension.channels = channels;
	this->tensorDimension.height = height;
	this->tensorDimension.width = width;
}

void Tensor::SetTensorData(float* tensor)
{
	this->tensorData = tensor;
}

void Tensor::SetTensorDimension(TensorDimension tensorDimension)
{
	this->tensorDimension = tensorDimension;
}


float * Tensor::GetTensorData() const
{
	return this->tensorData;
}

TensorDimension Tensor::GetTensorDimension()
{
	return this->tensorDimension;
}

#include "TensorDimension.h"

TensorDimension::TensorDimension()
{
}

TensorDimension::TensorDimension(int batchDimension, int channels, int height, int width)
{
	this->batchDimension = batchDimension;
	this->channels = channels;
	this->height = height;
	this->width = width;
	this->size = batchDimension * channels * height * width;
}

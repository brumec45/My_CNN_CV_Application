#pragma once
struct TensorDimension
{
	int batchDimension;
	int channels;
	int height;
	int width;
	int size;

	TensorDimension();
	TensorDimension(int batchDimension, int channels, int height, int width);
};
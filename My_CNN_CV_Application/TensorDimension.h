#pragma once
struct TensorDimension
{
	int batchDimension;
	int channels;
	int height;
	int width;

	TensorDimension();
	TensorDimension(int batchDimension, int channels, int height, int width);
};
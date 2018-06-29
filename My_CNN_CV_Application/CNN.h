#pragma once
#include <vector>
#include "Conv_Layer_GPU.h"
#include "MaxPool_Layer_GPU.h"

class CNN {
private:
	std::vector<Layer*> layers;

	void AddConvLayer(const TensorDimension &firstLayerInputTensorDimension, const TensorDimension &kernalTensorDimension, const TensorDimension &outputTensorDimension);
	void AddConvLayer(const TensorDimension &kernalTensorDimension, const TensorDimension &outputTensorDimension);
	void AddMaxPoolLayer(const TensorDimension &inputTensorDimension, const TensorDimension &outputTensorDimension, int stride);
	void AddMaxPoolLayer(const TensorDimension &outputTensorDimension, int stride);
	void AddConv_MaxPool_Combo(int outputSize, int featureDepth);

public:
	CNN();
	void InitCNN();

};
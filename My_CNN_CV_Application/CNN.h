#pragma once
#include <vector>
#include "Conv_Layer_GPU.h"
class CNN {
private:
	std::vector<Layer*> layers;

	void AddConvLayer(const TensorDimension &firstLayerInputTensorDimension, const TensorDimension &kernalTensorDimension, const TensorDimension &outputTensorDimension);
	void AddConvLayer(const TensorDimension &kernalTensorDimension, const TensorDimension &outputTensorDimension);
public:
	CNN();
	void InitCNN();

};
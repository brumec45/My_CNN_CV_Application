#pragma once
#include <vector>
#include "Conv_Layer_GPU.h"
#include "MaxPool_Layer_GPU.h"

class CNN {
private:
	std::vector<Layer*> layers;

	void AddConvLayer(TensorDimension firstLayerInputTensorDimension, TensorDimension kernalTensorDimension, TensorDimension outputTensorDimension, int stride, int padding, int activationType);
	void AddConvLayer(TensorDimension kernalTensorDimension, TensorDimension outputTensorDimension, int stride, int padding, int activationType);
	void AddMaxPoolLayer(TensorDimension inputTensorDimension, TensorDimension outputTensorDimension, int stride, int padding);
	void AddMaxPoolLayer(TensorDimension outputTensorDimension, int stride, int padding);
	void AddConv_MaxPool_Combo(int outputSize, int featureDepth, int activationType);
	
public:
	CNN();
	void InitCNN();
	void GenerateTinyYOLOv2Architecture(int inputSize);
	void SetInput(float* input);
	void ForwardPass();
	float* GetOutput();
};
#pragma once
#include <vector>
#include <string>
#include "Conv_Layer_GPU.h"
#include "MaxPool_Layer_GPU.h"
#include "WeightsReader.h"

class CNN {
private:

	WeightsReader weightsReader;
	std::vector<Layer*> layers;

	void AddConvLayer(TensorDimension firstLayerInputTensorDimension, TensorDimension kernalTensorDimension, TensorDimension outputTensorDimension, int stride, int padding, int activationType, bool batchNormalization);
	void AddConvLayer(TensorDimension kernalTensorDimension, TensorDimension outputTensorDimension, int stride, int padding, int activationType, bool batchNormalization);
	void AddMaxPoolLayer(TensorDimension inputTensorDimension, TensorDimension outputTensorDimension, int stride, int padding);
	void AddMaxPoolLayer(TensorDimension outputTensorDimension, int stride, int padding);
	void AddConv_MaxPool_Combo(int outputSize, int featureDepth, int activationType, bool batchNormalization);
	
public:
	CNN();
	void InitCNN();
	void GenerateTinyYOLOv2Architecture(int inputSize, char* weightsFileName, bool batchNormalization);
	void GenerateSimpleConvolutionTestArchitecture(int inputSize, char* weightsFileName, bool batchNormalization);
	void SetInput(float* input);
	void ForwardPass();
	float* GetOutput();
};
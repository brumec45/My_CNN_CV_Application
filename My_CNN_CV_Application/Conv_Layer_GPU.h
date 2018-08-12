#pragma once
#include "Layer_GPU.h"
#include "BatchNormalization_Layer_GPU.h"

class Conv_Layer_GPU : public Layer_GPU {
private:
	Tensor *kernelTensor;
	BatchNormalization_Layer_GPU batchNormalizationLayer;

	int activationType; //0 - Leaky RELU, 1 - RELU
	bool batchNormalization = true;
	cudnnFilterDescriptor_t	     cudnnKernelDesc;
	cudnnConvolutionDescriptor_t cudnnConvDesc;

	int layerKernelBytes = 0;
	//Vsaka plast ima v GPU rezerviran spomin
	float *kernelGPUMemoryPointer = nullptr;

	void SetupCUDNN_Convolution();
	void AllocateCUDAMemory_Convolution();
public:
	Conv_Layer_GPU();
	Conv_Layer_GPU(Tensor *inputTensor, Tensor *outputTensor, Tensor *kernelTensor, int stride, int padding, int activationType, bool batchNormalization);

	void SetupCUDNN(bool firstLayer);
	void SetBatchNormalizationParameters(float * bnBias, float * bnScales, float * estimatedMean, float * estimatedVariance);
	Tensor* GetKernelTensor();
	float* GetKernelTensorWeights();
	void Forward();
};
#pragma once
#include "Conv_Layer.h"
#include "cuda.h"

class Conv_Layer_GPU : public Conv_Layer {
private:
	cudnnHandle_t                cudnnHandle;
	cudnnTensorDescriptor_t	     cudnnInputDesc;
	cudnnFilterDescriptor_t	     cudnnKernelDesc;
	cudnnTensorDescriptor_t	     cudnnOutputDesc;
	cudnnConvolutionDescriptor_t cudnnConvDesc;

	size_t forwardWorkspaceBytes = 0;
	int layerOutputBytes = 0, layerKernelBytes = 0;
	//Vsaka plast ima v GPU rezerviran spomin
	float *forwardWorkspaceGPUMemoryPointer = nullptr, *kernelGPUMemoryPointer = nullptr, *outputGPUMemoryPointer = nullptr;
public:
	Conv_Layer_GPU();
	Conv_Layer_GPU(Tensor inputTensor, Tensor outputTensor, Tensor kernelTensor);
	void SetupCUDNN();
	//Za naslednjo plast input
	float * GetOutputGPUMemoryPointer();
};
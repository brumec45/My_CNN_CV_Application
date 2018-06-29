#pragma once
#include "Layer_GPU.h"
#include "cuda.h"

class Conv_Layer_GPU : public Layer_GPU {
private:
	Tensor *kernelTensor;

	cudnnFilterDescriptor_t	     cudnnKernelDesc;
	cudnnConvolutionDescriptor_t cudnnConvDesc;

	int layerKernelBytes = 0;
	//Vsaka plast ima v GPU rezerviran spomin
	float *kernelGPUMemoryPointer = nullptr;

	void SetupCUDNN_Convolution();
	void AllocateCUDAMemory_Convolution();
public:
	Conv_Layer_GPU();
	Conv_Layer_GPU(Tensor *inputTensor, Tensor *outputTensor, Tensor *kernelTensor);

	void SetupCUDNN(bool firstLayer);
	void Forward();
};
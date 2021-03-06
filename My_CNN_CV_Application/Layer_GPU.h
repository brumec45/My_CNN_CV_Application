#pragma once
#include "Layer.h"
#include "cuda.h"
#include "RELU_Activation_Kernel.cuh"
#include "TensorSum_Kernel.cuh"
#include "Output_Layer_GPU_Kernels.cuh"

class Layer_GPU : public Layer {
protected:
	
	cudnnHandle_t                cudnnHandle;
	cudnnTensorDescriptor_t	     cudnnInputDesc;
	cudnnTensorDescriptor_t	     cudnnOutputDesc;
	const float alpha = 1, beta = 0;
	size_t forwardWorkspaceBytes = 0;

	int layerInputBytes = 0, layerOutputBytes = 0;
	//Vsaka plast ima v GPU rezerviran spomin
	float* forwardWorkspaceGPUMemoryPointer = nullptr, *inputGPUMemoryPointer = nullptr,
		 *outputGPUMemoryPointer = nullptr;

	void SetupCUDNNDescriptors_InputOutput();
	void AllocateCUDAMemory_InputOutput(bool firstLayer);
		
public:
	Layer_GPU();
	Layer_GPU(Tensor* inputTensor, Tensor* outputTensor);
	Layer_GPU(Tensor* inputTensor, Tensor* outputTensor, int stride, int padding);
			
	void SetInputData(float* inputData);

	//Za naslednjo plast input
	float* GetOutputGPUMemoryPointer();
	float* GetLayerOutputData();

};
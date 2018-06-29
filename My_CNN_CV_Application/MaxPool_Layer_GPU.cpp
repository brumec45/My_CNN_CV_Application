#include "MaxPool_Layer_GPU.h"

MaxPool_Layer_GPU::MaxPool_Layer_GPU()
{
}

MaxPool_Layer_GPU::MaxPool_Layer_GPU(Tensor * inputTensor, Tensor * outputTensor, int stride) : Layer_GPU(inputTensor, outputTensor)
{
	this->stride = stride;
}

void MaxPool_Layer_GPU::SetupCUDNN(bool firstLayer)
{
	checkCUDNN(cudnnCreate(&cudnnHandle));
	SetupCUDNNDescriptors_InputOutput();
	SetupCUDNN_MaxPooling();
	AllocateCUDAMemory_InputOutput(firstLayer);
}

void MaxPool_Layer_GPU::SetupCUDNN_MaxPooling()
{
	checkCUDNN(cudnnCreatePoolingDescriptor(&cudnnPoolingDesc));
	checkCUDNN(cudnnSetPooling2dDescriptor(cudnnPoolingDesc,
		cudnnPoolingMode,
		cudnnPoolingNanPropagation,
		/*window height*/2,
		/*window width*/ 2,
		/*vertical padding*/ this->padding,
		/*horizontal padding*/ this->padding,
		/*vertical stride*/ this->stride,
		/*horizontal stride*/ this->stride));
}


void MaxPool_Layer_GPU::Forward()
{
	checkCUDNN(cudnnPoolingForward(cudnnHandle,
		cudnnPoolingDesc,
		&alpha,
		cudnnInputDesc,
		this->inputGPUMemoryPointer,
		&beta,
		cudnnOutputDesc,
		this->outputGPUMemoryPointer));
}

int MaxPool_Layer_GPU::GetStride()
{
	return this->stride;
}
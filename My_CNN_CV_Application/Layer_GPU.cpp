#include "Layer.h"
#include "Layer_GPU.h"

Layer_GPU::Layer_GPU()
{
}

Layer_GPU::Layer_GPU(Tensor * inputTensor, Tensor * outputTensor, int stride, int padding) : Layer(inputTensor, outputTensor, stride, padding)
{
}

void Layer_GPU::SetupCUDNNDescriptors_InputOutput()
{
	checkCUDNN(cudnnCreateTensorDescriptor(&cudnnInputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(cudnnInputDesc,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/this->inputTensor->GetBatchDimension(),
		/*channels=*/this->inputTensor->GetChannels(),
		/*image_height=*/this->inputTensor->GetHeight(),
		/*image_width=*/this->inputTensor->GetWidth()));

	checkCUDNN(cudnnCreateTensorDescriptor(&cudnnOutputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(cudnnOutputDesc,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/this->outputTensor->GetBatchDimension(),
		/*channels=*/this->outputTensor->GetChannels(),
		/*image_height=*/this->outputTensor->GetHeight(),
		/*image_width=*/this->outputTensor->GetWidth()));
}

void Layer_GPU::AllocateCUDAMemory_InputOutput(bool firstLayer)
{
	cudaMalloc(&this->forwardWorkspaceGPUMemoryPointer, this->forwardWorkspaceBytes);

	this->layerInputBytes = this->inputTensor->GetBatchDimension() *
		this->inputTensor->GetChannels() *
		this->inputTensor->GetHeight() *
		this->inputTensor->GetWidth() * sizeof(float);

	//ce je prva plast nastavim nov input
	if (firstLayer)
	{
		cudaMalloc(&this->inputGPUMemoryPointer, this->layerInputBytes);
	}
	else
	{
		//input je enak prejsnjemu outputu
		this->inputGPUMemoryPointer = this->previousLayer->GetOutputGPUMemoryPointer();
	}

	this->layerOutputBytes = this->outputTensor->GetBatchDimension() *
		this->outputTensor->GetChannels() *
		this->outputTensor->GetHeight() *
		this->outputTensor->GetWidth() * sizeof(float);

	cudaMalloc(&this->outputGPUMemoryPointer, this->layerOutputBytes);
	cudaMemset(this->outputGPUMemoryPointer, 0, this->layerOutputBytes);
}

void Layer_GPU::SetInputData(float * inputData)
{
	cudaMemcpy(this->inputGPUMemoryPointer, inputData, this->layerInputBytes, cudaMemcpyHostToDevice);
}

float * Layer_GPU::GetOutputGPUMemoryPointer()
{
	return this->outputGPUMemoryPointer;
}

float * Layer_GPU::GetLayerOutputData()
{
	cudaMemcpy(this->outputTensor->GetTensorData(), this->outputGPUMemoryPointer, this->layerOutputBytes, cudaMemcpyDeviceToHost);
	return this->outputTensor->GetTensorData();
}


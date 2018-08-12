#include "BatchNormalization_Layer_GPU.h"



BatchNormalization_Layer_GPU::BatchNormalization_Layer_GPU()
{
}

BatchNormalization_Layer_GPU::BatchNormalization_Layer_GPU(Tensor * inputTensor, Tensor * outputTensor) : Layer_GPU(inputTensor, outputTensor)
{
}

BatchNormalization_Layer_GPU::BatchNormalization_Layer_GPU(TensorDimension tensorDimension)
{
	this->inputTensor = new Tensor(tensorDimension);
	this->outputTensor = new Tensor(tensorDimension);
}

void BatchNormalization_Layer_GPU::SetupCUDNN_BatchNormalizationInference()
{
	checkCUDNN(cudnnCreateTensorDescriptor(&cudnnNormDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(cudnnNormDesc,
		/*dataType=*/CUDNN_TENSOR_NCHW,
		/*format=*/CUDNN_DATA_FLOAT,
		/*out_channels=*/1,
		/*in_channels=*/this->inputTensor->GetChannels(),
		/*kernel_height=*/1,
		/*kernel_width=*/1)
	);
}

void BatchNormalization_Layer_GPU::AllocateCUDAMemory_BatchNormalizationInference()
{
	this->batchNormalizationParameterBytes = this->inputTensor->GetChannels() * sizeof(float);
	cudaMalloc(&this->bnBiasGPUMemoryPointer, this->batchNormalizationParameterBytes);
	checkCUDA(cudaPeekAtLastError());
	cudaMemcpy(this->bnBiasGPUMemoryPointer, this->bnBias, this->batchNormalizationParameterBytes, cudaMemcpyHostToDevice);
	checkCUDA(cudaPeekAtLastError());

	cudaMalloc(&this->bnScalesGPUMemoryPointer, this->batchNormalizationParameterBytes);
	cudaMemcpy(this->bnScalesGPUMemoryPointer, this->bnScales, this->batchNormalizationParameterBytes, cudaMemcpyHostToDevice);
	checkCUDA(cudaPeekAtLastError());

	cudaMalloc(&this->estimatedMeanGPUMemoryPointer, this->batchNormalizationParameterBytes);
	cudaMemcpy(this->estimatedMeanGPUMemoryPointer, this->estimatedMean, this->batchNormalizationParameterBytes, cudaMemcpyHostToDevice);
	checkCUDA(cudaPeekAtLastError());

	cudaMalloc(&this->estimatedVarianceGPUMemoryPointer, this->batchNormalizationParameterBytes);
	cudaMemcpy(this->estimatedVarianceGPUMemoryPointer, this->estimatedVariance, this->batchNormalizationParameterBytes, cudaMemcpyHostToDevice);
	checkCUDA(cudaPeekAtLastError());
}

void BatchNormalization_Layer_GPU::SetInputOutputDimension(TensorDimension tensorDimension)
{
	this->inputTensor = new Tensor(tensorDimension);
	this->outputTensor = new Tensor(tensorDimension);
}

void BatchNormalization_Layer_GPU::SetBatchNormalizationParameters(float * bnBias, float * bnScales, float * estimatedMean, float * estimatedVariance)
{
	this->bnBias = bnBias;
	this->bnScales = bnScales;
	this->estimatedMean = estimatedMean;
	this->estimatedVariance = estimatedVariance;
}

void BatchNormalization_Layer_GPU::SetupCUDNN()
{
	checkCUDNN(cudnnCreate(&cudnnHandle));
	SetupCUDNNDescriptors_InputOutput();
	SetupCUDNN_BatchNormalizationInference();
	AllocateCUDAMemory_BatchNormalizationInference();
}

void BatchNormalization_Layer_GPU::Normalize(float * gpuMemoryLocationToNormalize)
{
	checkCUDNN(cudnnBatchNormalizationForwardInference(
		this->cudnnHandle,
		CUDNN_BATCHNORM_SPATIAL,
		&alpha,
		&beta,
		this->cudnnInputDesc,
		gpuMemoryLocationToNormalize,
		this->cudnnOutputDesc,
		gpuMemoryLocationToNormalize,
		this->cudnnNormDesc,
		this->bnScalesGPUMemoryPointer,
		this->bnBiasGPUMemoryPointer,
		this->estimatedMeanGPUMemoryPointer,
		this->estimatedVarianceGPUMemoryPointer,
		this->epsilon)
	);
}


void BatchNormalization_Layer_GPU::Forward()
{
}

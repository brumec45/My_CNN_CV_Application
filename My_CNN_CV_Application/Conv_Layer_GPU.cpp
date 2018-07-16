#include "Conv_Layer_GPU.h"


Conv_Layer_GPU::Conv_Layer_GPU()
{
}

Conv_Layer_GPU::Conv_Layer_GPU(Tensor *inputTensor, Tensor *outputTensor, Tensor *kernelTensor, int stride, int padding) : Layer_GPU(inputTensor, outputTensor, stride, padding)
{
	this->kernelTensor = kernelTensor;
}


void Conv_Layer_GPU::SetupCUDNN(bool firstLayer)
{
	checkCUDNN(cudnnCreate(&cudnnHandle));
	SetupCUDNNDescriptors_InputOutput();
	SetupCUDNN_Convolution();
	AllocateCUDAMemory_InputOutput(firstLayer);
	AllocateCUDAMemory_Convolution();
	
}

void Conv_Layer_GPU::SetupCUDNN_Convolution()
{
	checkCUDNN(cudnnCreateFilterDescriptor(&cudnnKernelDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(cudnnKernelDesc,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/this->outputTensor->GetChannels(),
		/*in_channels=*/this->inputTensor->GetChannels(),
		/*kernel_height=*/this->kernelTensor->GetHeight(),
		/*kernel_width=*/this->kernelTensor->GetWidth()));
	
	checkCUDNN(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(cudnnConvDesc,
		/*pad_height=*/this->padding,
		/*pad_width=*/this->padding,
		/*vertical_stride=*/this->stride,
		/*horizontal_stride=*/this->stride,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_FLOAT));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		cudnnInputDesc,
		cudnnKernelDesc,
		cudnnConvDesc,
		cudnnOutputDesc,
		CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
		&this->forwardWorkspaceBytes));
}

void Conv_Layer_GPU::AllocateCUDAMemory_Convolution()
{
	this->layerKernelBytes = this->kernelTensor->GetBatchDimension() *
		this->kernelTensor->GetChannels() *
		this->kernelTensor->GetHeight() *
		this->kernelTensor->GetWidth() * sizeof(float);

	cudaMalloc(&this->kernelGPUMemoryPointer, this->layerKernelBytes);
	cudaMemcpy(this->kernelGPUMemoryPointer, this->kernelTensor->GetTensorData(), this->layerKernelBytes, cudaMemcpyHostToDevice);
}

void Conv_Layer_GPU::Forward()
{
	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
		&alpha,
		cudnnInputDesc,
		this->inputGPUMemoryPointer,
		cudnnKernelDesc,
		this->kernelGPUMemoryPointer,
		cudnnConvDesc,
		CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
		this->forwardWorkspaceGPUMemoryPointer,
		this->forwardWorkspaceBytes,
		&beta,
		cudnnOutputDesc,
		this->outputGPUMemoryPointer));
	
	LeakyRELUActivation(this->outputGPUMemoryPointer, this->outputTensor->GetTensorSize());
	checkCUDA(cudaPeekAtLastError());
}



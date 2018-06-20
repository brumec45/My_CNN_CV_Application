#include "Conv_Layer_GPU.h"

Conv_Layer_GPU::Conv_Layer_GPU()
{
}

Conv_Layer_GPU::Conv_Layer_GPU(Tensor inputTensor, Tensor outputTensor, Tensor filterTensor) : Conv_Layer(inputTensor, outputTensor, filterTensor)
{
}

void Conv_Layer_GPU::SetupCUDNN()
{
	checkCUDNN(cudnnCreateTensorDescriptor(&cudnnInputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(cudnnInputDesc,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/this->inputTensor.GetTensorDimension().batchDimension,
		/*channels=*/this->inputTensor.GetTensorDimension().channels,
		/*image_height=*/this->inputTensor.GetTensorDimension().height,
		/*image_width=*/this->inputTensor.GetTensorDimension().width));
		
	checkCUDNN(cudnnCreateTensorDescriptor(&cudnnOutputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(cudnnOutputDesc,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/this->outputTensor.GetTensorDimension().batchDimension,
		/*channels=*/this->outputTensor.GetTensorDimension().channels,
		/*image_height=*/this->outputTensor.GetTensorDimension().height,
		/*image_width=*/this->outputTensor.GetTensorDimension().width));
	
	checkCUDNN(cudnnCreateFilterDescriptor(&cudnnKernelDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(cudnnKernelDesc,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/this->outputTensor.GetTensorDimension().channels,
		/*in_channels=*/this->inputTensor.GetTensorDimension().channels,
		/*kernel_height=*/this->kernelTensor.GetTensorDimension().height,
		/*kernel_width=*/this->kernelTensor.GetTensorDimension().width));

	checkCUDNN(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(cudnnConvDesc,
		/*pad_height=*/1,
		/*pad_width=*/1,
		/*vertical_stride=*/1,
		/*horizontal_stride=*/1,
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
	
	//ves memory tudi na zacetku?
	cudaMalloc(&this->forwardWorkspaceGPUMemoryPointer, this->forwardWorkspaceBytes);

	//input je enak prejšnjemu outputu

	this->layerOutputBytes = this->outputTensor.GetTensorDimension().batchDimension *
		this->outputTensor.GetTensorDimension().channels *
		this->outputTensor.GetTensorDimension().height *
		this->outputTensor.GetTensorDimension().width;

	cudaMalloc(&this->outputGPUMemoryPointer, this->layerOutputBytes);
	cudaMemset(this->outputGPUMemoryPointer, 0, this->layerOutputBytes);

	this->layerKernelBytes = this->kernelTensor.GetTensorDimension().channels *
		this->kernelTensor.GetTensorDimension().height *
		this->kernelTensor.GetTensorDimension().width;

	cudaMalloc(&this->kernelGPUMemoryPointer, this->layerKernelBytes);
	cudaMemcpy(this->kernelGPUMemoryPointer, this->kernelTensor.GetTensorData(), this->layerKernelBytes, cudaMemcpyHostToDevice);
	
}

float * Conv_Layer_GPU::GetOutputGPUMemoryPointer()
{
	return this->outputGPUMemoryPointer;
}



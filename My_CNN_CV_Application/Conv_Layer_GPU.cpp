#include "Conv_Layer_GPU.h"

Conv_Layer_GPU::Conv_Layer_GPU()
{
}

Conv_Layer_GPU::Conv_Layer_GPU(Tensor *inputTensor, Tensor *outputTensor, Tensor *kernelTensor) : Conv_Layer(inputTensor, outputTensor, kernelTensor)
{
}


void Conv_Layer_GPU::SetupCUDNN(bool firstLayer)
{
	checkCUDNN(cudnnCreate(&cudnnHandle));

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

	this->layerInputBytes = this->inputTensor->GetBatchDimension() *
		this->inputTensor->GetChannels() *
		this->inputTensor->GetHeight() *
		this->inputTensor->GetWidth() * sizeof(float);
		

	//ce je prva plast nastavim nov input
	if (firstLayer)
	{
		cudaMalloc(&this->inputGPUMemoryPointer, this->layerInputBytes);
		//cudaMemcpy(this->inputGPUMemoryPointer, this->inputTensor.GetTensorData(), this->layerInputBytes, cudaMemcpyHostToDevice);
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

	/*const float kernel_template[3][3] = {
		{ 1,  1, 1 },
		{ 1, -8, 1 },
		{ 1,  1, 1 }
	};

	float h_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel) {
		for (int channel = 0; channel < 3; ++channel) {
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}*/


	this->layerKernelBytes = this->kernelTensor->GetBatchDimension() *
		this->kernelTensor->GetChannels() *
		this->kernelTensor->GetHeight() *
		this->kernelTensor->GetWidth() * sizeof(float);

	/*float t = 0;
	for (size_t i = 0; i < 81; i++)
	{
		t = this->kernelTensor.GetTensorData()[i];

	}*/

	cudaMalloc(&this->kernelGPUMemoryPointer, this->layerKernelBytes);
	cudaMemcpy(this->kernelGPUMemoryPointer, this->kernelTensor->GetTensorData(), this->layerKernelBytes, cudaMemcpyHostToDevice);
	
}

void Conv_Layer_GPU::Forward()
{
	const float alpha = 1, beta = 0;
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
}

void Conv_Layer_GPU::SetInputData(float * inputData)
{
	cudaMemcpy(this->inputGPUMemoryPointer, inputData, this->layerInputBytes, cudaMemcpyHostToDevice);
}

float * Conv_Layer_GPU::GetOutputGPUMemoryPointer()
{
	return this->outputGPUMemoryPointer;
}

float * Conv_Layer_GPU::GetLayerOutputData()
{
	float* outputData = new float[this->layerOutputBytes];
	cudaMemcpy(outputData, this->outputGPUMemoryPointer, this->layerOutputBytes, cudaMemcpyDeviceToHost);
	return outputData;
}



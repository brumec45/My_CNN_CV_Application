#include "TensorSum_Kernel.cuh"


__global__ void AddBIAS_Kernel(float* tensor, float* bias, int biasSize, int tensorHeight, int tensorWidth)
{
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int tensorXYSize = tensorHeight * tensorWidth;
	int tensorSize = biasSize * tensorXYSize;

	if (threadIndex < tensorSize)
	{
		int threadDepthIndex = threadIndex % biasSize;
		int threadXYIndex = threadIndex % tensorXYSize;

		//tensor[threadDepthIndex * tensorXYSize + threadXYIndex] = threadDepthIndex;
		tensor[threadDepthIndex * tensorXYSize + threadXYIndex] += bias[threadDepthIndex];
	}
}


void AddBIAS_GPU(float* tensor, float* bias, int biasSize, int tensorHeight, int tensorWidth) {
	
	int tensorSize = biasSize * tensorHeight * tensorWidth;
	int gridXDim = ceil(tensorSize / 512.0);
	AddBIAS_Kernel << <gridXDim, 512 >> > (tensor, bias, biasSize, tensorHeight, tensorWidth);
}


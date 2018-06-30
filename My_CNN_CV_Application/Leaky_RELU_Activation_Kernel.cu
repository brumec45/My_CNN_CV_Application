#include "Leaky_RELU_Activation_Kernel.cuh"

__global__ void Leaky_RELU_Kernel(float* tensor, int tensorSize)
{
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (threadIndex < tensorSize)
	{
		if (tensor[threadIndex] > 0)
		{
			tensor[threadIndex] = tensor[threadIndex];
		}
		else
		{
			tensor[threadIndex] = tensor[threadIndex] * 0.1f;
		}
	}
}

void LeakyRELUActivation(float* tensor, int tensorSize) {
	//vedno je deljivo z 512 (4)
	Leaky_RELU_Kernel << <tensorSize / 512, 512 >> > (tensor, tensorSize);
}
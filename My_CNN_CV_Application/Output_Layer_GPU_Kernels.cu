#include "Output_Layer_GPU_Kernels.cuh"

__constant__ float anchors_416[10] = { 1.08, 1.19,  3.42, 4.41,  6.63, 11.38,  9.42, 5.11,  16.62, 10.52 };

__device__ float Sigmoid(float x) 
{
	float expValue = exp((double)-x);
	float result = 1 / (1 + expValue);

	return result;
}

__global__ void XY_BoundingBox_Coordinates_Transform_Kernel(float* input, int inputHeight, int inputWidth) 
{
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int tensorXYSize = inputHeight * inputWidth;
	int tensorSize = XYCoordinatesCount * tensorXYSize;
	

	if (threadIndex < tensorSize)
	{
		int threadDepthIndex = threadIndex % XYCoordinatesCount;
		//int threadDepthIndexY = (threadIndex % XYCoordinatesCount) + 1;
		int threadXYIndex = threadIndex % tensorXYSize;
		int cy = threadXYIndex / inputWidth;
		int cx = threadXYIndex % inputWidth;
				
		//tensor[threadDepthIndex * tensorXYSize + threadXYIndex] = threadDepthIndex;
		input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex] = (cx + Sigmoid(input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex])) * downsampleFactor;
		input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex] = (cy + Sigmoid(input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex])) * downsampleFactor;
		//input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex] = threadDepthIndex * 4 * tensorXYSize + threadXYIndex;
		//input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex] = (threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex;
		//input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex] = cx;
		//input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex] = cy;
	}
}
__global__ void WH_BoundingBox_Transform_Kernel(float* input, int inputHeight, int inputWidth) 
{
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int tensorXYSize = inputHeight * inputWidth;
	int tensorSize = WHCoordinatesCount * tensorXYSize;

	if (threadIndex < tensorSize)
	{
		int threadDepthIndex = threadIndex % XYCoordinatesCount;
		//int threadDepthIndexY = (threadIndex % XYCoordinatesCount) + 1;
		int threadXYIndex = threadIndex % tensorXYSize;
		int cy = threadXYIndex / inputWidth;
		int cx = threadXYIndex % inputWidth;

		//tensor[threadDepthIndex * tensorXYSize + threadXYIndex] = threadDepthIndex;
		input[(threadDepthIndex * 4 + 2) * tensorXYSize + threadXYIndex] = exp(input[(threadDepthIndex * 4 + 2) * tensorXYSize + threadXYIndex]) *
			anchors_416[2 * threadDepthIndex] * downsampleFactor;
		input[(threadDepthIndex * 4 + 3) * tensorXYSize + threadXYIndex] = exp(input[(threadDepthIndex * 4 + 3) * tensorXYSize + threadXYIndex]) *
			anchors_416[2 * threadDepthIndex + 1] * downsampleFactor;
		//input[(threadDepthIndex * 4 + 2) * tensorXYSize + threadXYIndex] = anchors_416[2 * threadDepthIndex];
		//input[(threadDepthIndex * 4 + 3) * tensorXYSize + threadXYIndex] = anchors_416[2 * threadDepthIndex + 1];
	}

}


void WH_BoundingBox_Transform(float* input, int inputHeight, int inputWidth) 
{
	int WHCoordinatesCount = 5;
	int tensorSize = WHCoordinatesCount * inputHeight * inputWidth;
	int gridXDim = ceil(tensorSize / 512.0);
	WH_BoundingBox_Transform_Kernel << <gridXDim, 512 >> > (input, inputHeight, inputWidth);
}

void Output_Transform_GPU(float* input, int inputHeight, int inputWidth)
{
	int XYCoordinatesCount = 5;
	int tensorSize = XYCoordinatesCount * inputHeight * inputWidth;
	int gridXDim = ceil(tensorSize / 512.0);
	XY_BoundingBox_Coordinates_Transform_Kernel << <gridXDim, 512 >> > (input, inputHeight, inputWidth);
}
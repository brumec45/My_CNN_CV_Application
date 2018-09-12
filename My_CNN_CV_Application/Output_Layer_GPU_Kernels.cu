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
	int tensorSize = boundingBoxesPerGridCell * tensorXYSize;
	

	if (threadIndex < tensorSize)
	{
		int threadDepthIndex = threadIndex % boundingBoxesPerGridCell;
		//int threadDepthIndexY = (threadIndex % XYCoordinatesCount) + 1;
		int threadXYIndex = threadIndex % tensorXYSize;
		int cy = threadXYIndex / inputWidth;
		int cx = threadXYIndex % inputWidth;
				
		//tensor[threadDepthIndex * tensorXYSize + threadXYIndex] = threadDepthIndex;
		input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex] = (cx + Sigmoid(input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex])) * downsampleFactor;
		input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex] = (cy + Sigmoid(input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex])) * downsampleFactor;
		//input[threadDepthIndex * 4 * tensorXYSize + threadXYIndex] = 1;
		//input[(threadDepthIndex * 4 + 1) * tensorXYSize + threadXYIndex] = 1;
	}
}
__global__ void WH_BoundingBox_Transform_Kernel(float* input, int inputHeight, int inputWidth) 
{
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int tensorXYSize = inputHeight * inputWidth;
	int tensorSize = boundingBoxesPerGridCell * tensorXYSize;

	if (threadIndex < tensorSize)
	{
		int threadDepthIndex = threadIndex % boundingBoxesPerGridCell;
		//int threadDepthIndexY = (threadIndex % XYCoordinatesCount) + 1;
		int threadXYIndex = threadIndex % tensorXYSize;
		
		//tensor[threadDepthIndex * tensorXYSize + threadXYIndex] = threadDepthIndex;
		input[(threadDepthIndex * 4 + 2) * tensorXYSize + threadXYIndex] = exp(input[(threadDepthIndex * 4 + 2) * tensorXYSize + threadXYIndex]) *
			anchors_416[2 * threadDepthIndex] * downsampleFactor;
		input[(threadDepthIndex * 4 + 3) * tensorXYSize + threadXYIndex] = exp(input[(threadDepthIndex * 4 + 3) * tensorXYSize + threadXYIndex]) *
			anchors_416[2 * threadDepthIndex + 1] * downsampleFactor;
		//input[(threadDepthIndex * 4 + 2) * tensorXYSize + threadXYIndex] = anchors_416[2 * threadDepthIndex] = 1;
		//input[(threadDepthIndex * 4 + 3) * tensorXYSize + threadXYIndex] = anchors_416[2 * threadDepthIndex + 1] = 1;

		input[(20 + threadDepthIndex) * tensorXYSize + threadXYIndex] = Sigmoid(input[(20 + threadDepthIndex) * tensorXYSize + threadXYIndex]);
		//input[(20 + threadDepthIndex) * tensorXYSize + threadXYIndex] = 2;
	}
}

__global__ void Softmax_Kernel(float* input, int classesCount, int inputHeight, int inputWidth)
{
	int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int tensorXYSize = inputHeight * inputWidth;
	int tensorSize = boundingBoxesPerGridCell * tensorXYSize;

	if (threadIndex < tensorSize)
	{
		int threadDepthIndex = threadIndex % boundingBoxesPerGridCell;
		int threadXYIndex = threadIndex % tensorXYSize;
		float maxClassProbability = FLOAT_MIN;

		for (size_t i = 0; i < classesCount; i++)
		{
			float classProbability = input[(25 + threadDepthIndex * classesCount + i) * tensorXYSize + threadXYIndex];

			if (classProbability > maxClassProbability)
			{
				maxClassProbability = classProbability;
			}
		}

		float classProbabilitiesSum = 0;
		for (size_t i = 0; i < classesCount; i++)
		{
			float exponent = exp(input[(25 + threadDepthIndex * classesCount + i) * tensorXYSize + threadXYIndex] - maxClassProbability);
			classProbabilitiesSum += exponent;
			input[(25 + threadDepthIndex * classesCount + i) * tensorXYSize + threadXYIndex] = exponent;
		}

		for (size_t i = 0; i < classesCount; i++)
		{
			input[(25 + threadDepthIndex * classesCount + i) * tensorXYSize + threadXYIndex] /= classProbabilitiesSum;
			//input[(25 + threadDepthIndex * classesCount + i) * tensorXYSize + threadXYIndex] = i;
			//input[(25 + threadDepthIndex * classesCount + i) * tensorXYSize + threadXYIndex] = 3;
		}
	}
}


void WH_BoundingBox_Transform(float* input, int inputHeight, int inputWidth) 
{
	int tensorSize = boundingBoxesPerGridCell * inputHeight * inputWidth;
	int gridXDim = ceil(tensorSize / 512.0);
	WH_BoundingBox_Transform_Kernel << <gridXDim, 512 >> > (input, inputHeight, inputWidth);
}

void XY_BoundingBox_Coordinates_Transform(float* input, int inputHeight, int inputWidth)
{
	int tensorSize = boundingBoxesPerGridCell * inputHeight * inputWidth;
	int gridXDim = ceil(tensorSize / 512.0);
	XY_BoundingBox_Coordinates_Transform_Kernel << <gridXDim, 512 >> > (input, inputHeight, inputWidth);
}

void Softmax_GPU(float* input, int classesCount, int inputHeight, int inputWidth)
{
	int tensorSize = boundingBoxesPerGridCell * inputHeight * inputWidth;
	int gridXDim = ceil(tensorSize / 512.0);
	Softmax_Kernel << <gridXDim, 512 >> > (input, classesCount, inputHeight, inputWidth);
}
#pragma once
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <math.h>

__device__ const int downsampleFactor = 32;
__device__ const int boundingBoxesPerGridCell = 5;
__device__ const float FLOAT_MIN = 1.17549435e-38;

//__global__ void BoundingBox_ConfidenceScores_Transform_Kernel(float* input, int inputHeight, int inputWidth);
__global__ void XY_BoundingBox_Coordinates_Transform_Kernel(float* input, int inputHeight, int inputWidth);
__global__ void WH_BoundingBox_Transform_Kernel(float* input, int inputHeight, int inputWidth);
__global__ void Softmax_Kernel(float* input, int classesCount, int inputHeight, int inputWidth);

void XY_BoundingBox_Coordinates_Transform(float* input, int inputHeight, int inputWidth);
void WH_BoundingBox_Transform(float* input, int inputHeight, int inputWidth);
void Softmax_GPU(float* input, int classesCount, int inputHeight, int inputWidth);
//void BoundingBox_ConfidenceScores_Transform(float* input, int inputHeight, int inputWidth);
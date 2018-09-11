#pragma once
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <math.h>

const int downsampleFactor = 32;
const int XYCoordinatesCount = 5;
const int WHCoordinatesCount = 5;

__global__ void BoundingBox_ConfidenceScores_Transform_Kernel(float* input, int inputHeight, int inputWidth);
__global__ void XY_BoundingBox_Coordinates_Transform_Kernel(float* input, int inputHeight, int inputWidth);
__global__ void WH_BoundingBox_Transform_Kernel(float* input, int inputHeight, int inputWidth);

void Output_Transform_GPU(float* input, int inputHeight, int inputWidth);
void WH_BoundingBox_Transform(float* input, int inputHeight, int inputWidth);
void BoundingBox_ConfidenceScores_Transform(float* input, int inputHeight, int inputWidth);
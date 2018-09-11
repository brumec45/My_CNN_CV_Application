#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<math.h>

__global__ void AddBIAS_Kernel(float* tensor, float* bias, int biasSize, int tensorHeight, int tensorWidth);
void AddBIAS_GPU(float* tensor, float* bias, int biasSize, int tensorHeight, int tensorWidth);



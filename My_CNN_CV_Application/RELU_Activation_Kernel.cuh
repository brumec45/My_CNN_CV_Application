#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void Leaky_RELU_Kernel(float* tensor, int tensorSize);
__global__ void RELU_Kernel(float* tensor, int tensorSize);
void LeakyRELUActivation(float* tensor, int tensorSize);
void RELUActivation(float* tensor, int tensorSize);


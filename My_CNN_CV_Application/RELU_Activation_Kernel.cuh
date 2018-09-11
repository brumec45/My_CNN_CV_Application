#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void Leaky_RELU_Kernel(float* tensor, int tensorSize);
__global__ void RELU_Kernel(float* tensor, int tensorSize);
void LeakyRELUActivation_GPU(float* tensor, int tensorSize);
void RELUActivation_GPU(float* tensor, int tensorSize);


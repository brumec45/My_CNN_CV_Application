#pragma once
#include <cudnn.h>
#include <cuda.h>
#include <iostream>
void checkCUDNN(cudnnStatus_t status);
void checkCUDA(cudaError_t code);
void CUDADeviceInfo();

#include "cuda.h"

void checkCUDNN(cudnnStatus_t status)                               
{                                                          
	if (status != CUDNN_STATUS_SUCCESS)
	{
		std::cout << "Error on line " << __LINE__ << ": "
			<< cudnnGetErrorString(status) << std::endl;
		std::getchar();
		std::exit(EXIT_FAILURE);
	}
}

void checkCUDA(cudaError_t code)
{
	if (code != cudaSuccess)
	{
		std::cout << "Error on line " << __LINE__ << ": "
			<< cudaGetErrorString(code) << std::endl;
		std::getchar();
		std::exit(EXIT_FAILURE);
		
	}
}

void CUDADeviceInfo()
{
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("  Max threads dim: %d\n", prop.maxThreadsDim);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}

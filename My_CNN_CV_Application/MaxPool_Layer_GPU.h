#pragma once
#include "Layer_GPU.h"

class MaxPool_Layer_GPU: public Layer_GPU {
private:
	cudnnPoolingDescriptor_t cudnnPoolingDesc;
	const cudnnPoolingMode_t cudnnPoolingMode = CUDNN_POOLING_MAX;
	const cudnnNanPropagation_t cudnnPoolingNanPropagation = CUDNN_NOT_PROPAGATE_NAN;

	int windowHeight = 2, windowWidth = 2;

	void SetupCUDNN_MaxPooling();
public:
	MaxPool_Layer_GPU();
	MaxPool_Layer_GPU(Tensor *inputTensor, Tensor *outputTensor, int stride, int padding);
		
	void SetupCUDNN(bool firstLayer);
	void Forward();

	int GetStride();

};
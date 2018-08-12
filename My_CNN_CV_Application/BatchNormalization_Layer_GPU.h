#pragma once
#include "Layer_GPU.h"

class BatchNormalization_Layer_GPU : public Layer_GPU {
private:

	cudnnTensorDescriptor_t	     cudnnNormDesc;

	float *bnBias, *bnScales, *estimatedMean, *estimatedVariance;
	double epsilon = 0.00001;

	float * bnBiasGPUMemoryPointer = nullptr, *bnScalesGPUMemoryPointer = nullptr, *estimatedMeanGPUMemoryPointer = nullptr,
	*estimatedVarianceGPUMemoryPointer = nullptr;

	int batchNormalizationParameterBytes;

	void SetupCUDNN_BatchNormalizationInference();
	void AllocateCUDAMemory_BatchNormalizationInference();
public:
	BatchNormalization_Layer_GPU();
	BatchNormalization_Layer_GPU(Tensor* inputTensor, Tensor* outputTensor);
	BatchNormalization_Layer_GPU(TensorDimension tensorDimension);

	void SetInputOutputDimension(TensorDimension tensorDimension);
	void SetBatchNormalizationParameters(float *bnBias, float *bnScales, float *estimatedMean, float *estimatedVariance);
	void SetupCUDNN();
	void Normalize(float *gpuMemoryLocationToNormalize);
	void Forward();
};


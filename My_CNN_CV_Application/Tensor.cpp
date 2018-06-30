#include "Tensor.h"

Tensor::Tensor()
{
}

Tensor::Tensor(const TensorDimension &tensorDimension)
{
	this->tensorSize = tensorDimension.batchDimension * tensorDimension.channels * tensorDimension.height * tensorDimension.width;
	this->tensorData = new float[this->tensorSize];
	this->tensorDimension = tensorDimension;

}

void Tensor::InitKernelWeights()
{
	
	//test
	float kernel_template[3][3] = {
		{ 1,  1, 1 },
		{ 1, -8, 1 },
		{ 1,  1, 1 }
	};
	
	int kernel2DSize = this->tensorDimension.height *
		this->tensorDimension.width;

	int kernel3DSize = this->tensorDimension.channels * kernel2DSize;
	//this->tensorData[0] = 1.0;
	//za vsak filter
	for (int bd = 0; bd < this->tensorDimension.batchDimension; bd++)
	{
		int kernelStartIndex = bd * kernel3DSize;
		for (int c = 0; c < this->tensorDimension.channels; c++)
		{
			int channelStartIndex = kernelStartIndex + c * kernel2DSize;
			for (int h = 0; h < this->tensorDimension.height; h++)
			{
				int heightIndex = channelStartIndex + h * this->tensorDimension.width;
				for (int w = 0; w < this->tensorDimension.width; w++)
				{
					this->tensorData[heightIndex + w] = kernel_template[h][w];
				}
			}
		}
	}
}



void Tensor::SetTensorData(float* tensor)
{
	this->tensorData = tensor;
}

void Tensor::SetTensorDimension(const TensorDimension &tensorDimension)
{
	this->tensorDimension = tensorDimension;
}


float * Tensor::GetTensorData()
{
	return this->tensorData;
}

int Tensor::GetBatchDimension()
{
	return this->tensorDimension.batchDimension;
}

int Tensor::GetChannels()
{
	return this->tensorDimension.channels;
}

int Tensor::GetHeight()
{
	return this->tensorDimension.height;
}

int Tensor::GetWidth()
{
	return this->tensorDimension.width;
}

int Tensor::GetTensorSize()
{
	return this->tensorSize;
}

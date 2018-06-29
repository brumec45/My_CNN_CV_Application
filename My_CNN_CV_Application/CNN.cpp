#include "CNN.h"
#include "ImageInput.h"

CNN::CNN()
{
}

void CNN::InitCNN()
{
	Mat testImage = LoadImage("more_of_lenna.jpg");
	Mat myBlob = cv::dnn::blobFromImage(testImage, 1, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
	
	float * myBlobF = myBlob.ptr<float>();
	int inputSize = 416;
	int featureDepth = 16;
	int outputSize = 416;

	TensorDimension inputTensor1(1, 3, inputSize, inputSize);
	TensorDimension kernelTensor1(featureDepth, 3, 3, 3);
	TensorDimension outputTensor1(1, featureDepth, outputSize, outputSize);
	AddConvLayer(inputTensor1, kernelTensor1, outputTensor1);
	
	TensorDimension outputTensor2(1, featureDepth, outputSize / 2, outputSize / 2);
	AddMaxPoolLayer(outputTensor2, 2);
		
	for (size_t i = 0; i < 4; i++)
	{
		outputSize /= 2;
		featureDepth *= 2;
		AddConv_MaxPool_Combo(outputSize, featureDepth);
	}
	outputSize /= 2;

	this->layers[0]->SetInputData(myBlobF);

	for (size_t i = 0; i < this->layers.size(); i++)
	{
		this->layers[i]->Forward();
	}
	

	float* outputData = this->layers[this->layers.size() - 1]->GetLayerOutputData();

	Mat result(outputSize, outputSize, CV_32FC1);
	//float* resultP = result.

	for (int i = 0; i < outputSize; i++)
	{
		//resultP = result.data + i * result.step;
		for (int j = 0; j < outputSize; j++)
		{
			//resultP[j] = 100;// h_output[i * 244 + j];
			result.at<float>(i, j) = outputData[i * outputSize + j];
		}
	}
	//float a = h_output[10];
	imshow("src", result);
	cv::waitKey(0);
}

void CNN::AddConv_MaxPool_Combo(int outputSize, int featureDepth) 
{
	TensorDimension kernelTensor1(featureDepth, featureDepth / 2, 3, 3);
	TensorDimension outputTensor1(1, featureDepth, outputSize, outputSize);
	AddConvLayer(kernelTensor1, outputTensor1);

	TensorDimension outputTensor2(1, featureDepth, outputSize / 2, outputSize / 2);
	AddMaxPoolLayer(outputTensor2, 2);
}

void CNN::AddConvLayer(const TensorDimension & firstLayerInputTensorDimension, const TensorDimension & kernalTensorDimension, const TensorDimension & outputTensorDimension)
{
	Tensor* inputTensor = new Tensor(firstLayerInputTensorDimension);
	
	//inputTensor.SetTensorData(myBlobF);
		
	Tensor* kernelTensor = new Tensor(kernalTensorDimension);
	kernelTensor->InitKernelWeights();
		
	Tensor* outputTensor = new Tensor(outputTensorDimension);

	Conv_Layer_GPU* convLayerGPU = new Conv_Layer_GPU(inputTensor, outputTensor, kernelTensor);

	convLayerGPU->SetPreviousLayer(nullptr);
	convLayerGPU->SetupCUDNN(true);

	this->layers.push_back(convLayerGPU);
}

void CNN::AddConvLayer(const TensorDimension & kernalTensorDimension, const TensorDimension & outputTensorDimension)
{
	if (this->layers.size() == 0)
	{
		return;
	}

	Tensor* inputTensor = this->layers[this->layers.size() - 1]->GetOutputTensor();
	//inputTensor.SetTensorData(myBlobF);

	Tensor* kernelTensor = new Tensor(kernalTensorDimension);
	kernelTensor->InitKernelWeights();

	Tensor* outputTensor = new Tensor(outputTensorDimension);

	Conv_Layer_GPU* convLayerGPU = new Conv_Layer_GPU(inputTensor, outputTensor, kernelTensor);

	convLayerGPU->SetPreviousLayer(this->layers[this->layers.size() - 1]);
	convLayerGPU->SetupCUDNN(false);

	this->layers.push_back(convLayerGPU);
}

void CNN::AddMaxPoolLayer(const TensorDimension & inputTensorDimension, const TensorDimension & outputTensorDimension, int stride)
{
}

void CNN::AddMaxPoolLayer(const TensorDimension & outputTensorDimension, int stride)
{
	if (this->layers.size() == 0)
	{
		return;
	}

	Tensor* inputTensor = this->layers[this->layers.size() - 1]->GetOutputTensor();
	Tensor* outputTensor = new Tensor(outputTensorDimension);

	MaxPool_Layer_GPU* maxPoolLayerGPU = new MaxPool_Layer_GPU(inputTensor, outputTensor, stride);
	maxPoolLayerGPU->SetPreviousLayer(this->layers[this->layers.size() - 1]);
	maxPoolLayerGPU->SetupCUDNN(false);
	this->layers.push_back(maxPoolLayerGPU);
}





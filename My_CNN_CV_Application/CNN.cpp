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
	
		
	/*Mat result(outputSize, outputSize, CV_32FC1);
	//float* resultP = result.
	float t;
	for (int i = 0; i < outputSize; i++)
	{
		//resultP = result.data + i * result.step;
		for (int j = 0; j < outputSize; j++)
		{
			//resultP[j] = 100;// h_output[i * 244 + j];
			t = outputData[i * outputSize + j];
			result.at<float>(i, j) = outputData[i * outputSize + j];
		}
	}*/
	//float a = h_output[10];
	//imshow("src", result);
	//cv::waitKey(0);
}

void CNN::GenerateTinyYOLOv2Architecture(int inputSize, char* weightsFileName, bool batchNormalization)
{
	this->weightsReader.OpenFile(weightsFileName);
	int featureDepth = 16;
	int outputSize = inputSize;

	TensorDimension inputTensor1(1, 3, inputSize, inputSize);
	TensorDimension kernelTensor1(featureDepth, 3, 3, 3);
	TensorDimension outputTensor1(1, featureDepth, outputSize, outputSize);
	AddConvLayer(inputTensor1, kernelTensor1, outputTensor1, 1, 1, 0, batchNormalization);

	TensorDimension outputTensor2(1, featureDepth, outputSize / 2, outputSize / 2);
	AddMaxPoolLayer(outputTensor2, 2, 0);

	for (size_t i = 0; i < 4; i++)
	{
		outputSize /= 2;
		featureDepth *= 2;
		AddConv_MaxPool_Combo(outputSize, featureDepth, 0, batchNormalization);
	}
	outputSize /= 2; //13x13
	featureDepth *= 2;//512

	TensorDimension kernelTensor3(featureDepth, featureDepth / 2, 3, 3);
	TensorDimension outputTensor3(1, featureDepth, outputSize, outputSize);
	AddConvLayer(kernelTensor3, outputTensor3, 1, 1, 0, batchNormalization);

	TensorDimension outputTensor4(1, featureDepth, outputSize, outputSize);
	AddMaxPoolLayer(outputTensor4, 1, 0);

	featureDepth *= 2;//1024

	TensorDimension kernelTensor4(featureDepth, featureDepth / 2, 3, 3);
	TensorDimension outputTensor5(1, featureDepth, outputSize, outputSize);
	AddConvLayer(kernelTensor4, outputTensor5, 1, 1, 0, batchNormalization);

	TensorDimension kernelTensor5(featureDepth, featureDepth, 3, 3);
	TensorDimension outputTensor6(1, featureDepth, outputSize, outputSize);
	AddConvLayer(kernelTensor5, outputTensor6, 1, 1, 0, batchNormalization);

	TensorDimension kernelTensor6(125, featureDepth, 1, 1);
	TensorDimension outputTensor7(1, 125, outputSize, outputSize);
	AddConvLayer(kernelTensor6, outputTensor7, 1, 0, 1, false);
	this->weightsReader.CloseFile();
}

void CNN::GenerateSimpleConvolutionTestArchitecture(int inputSize, char* weightsFileName, bool batchNormalization)
{
	this->weightsReader.OpenFile(weightsFileName);
	int featureDepth = 16;
	int outputSize = inputSize;

	TensorDimension inputTensor1(1, 3, inputSize, inputSize);
	TensorDimension kernelTensor1(featureDepth, 3, 3, 3);
	TensorDimension outputTensor1(1, featureDepth, outputSize, outputSize);
	AddConvLayer(inputTensor1, kernelTensor1, outputTensor1, 1, 1, 0, batchNormalization);
	this->weightsReader.CloseFile();
}

void CNN::SetInput(float * input)
{
	if (this->layers.size() == 0)
	{
		return;
	}

	this->layers[0]->SetInputData(input);
}

void CNN::ForwardPass()
{
	for (size_t i = 0; i < this->layers.size(); i++)
	{
		this->layers[i]->Forward();
	}
}

float * CNN::GetOutput()
{
	float* outputData = this->layers[this->layers.size() - 1]->GetLayerOutputData();
	return outputData;
}

void CNN::AddConv_MaxPool_Combo(int outputSize, int featureDepth, int activationType, bool batchNormalization)
{
	TensorDimension kernelTensor1(featureDepth, featureDepth / 2, 3, 3);
	TensorDimension outputTensor1(1, featureDepth, outputSize, outputSize);
	AddConvLayer(kernelTensor1, outputTensor1, 1, 1, activationType, batchNormalization);
	
	TensorDimension outputTensor2(1, featureDepth, outputSize / 2, outputSize / 2);
	AddMaxPoolLayer(outputTensor2, 2, 0);
}

void CNN::AddConvLayer(TensorDimension firstLayerInputTensorDimension, TensorDimension kernalTensorDimension, TensorDimension outputTensorDimension, int stride, int padding, int activationType, bool batchNormalization)
{
	Tensor* inputTensor = new Tensor(firstLayerInputTensorDimension);
	Tensor* kernelTensor = new Tensor(kernalTensorDimension);
	Tensor* outputTensor = new Tensor(outputTensorDimension);
	Conv_Layer_GPU* convLayerGPU = new Conv_Layer_GPU(inputTensor, outputTensor, kernelTensor, stride, padding, activationType, batchNormalization);

	int outputFeatureDepth = outputTensor->GetChannels();
	float * bias = new float[outputFeatureDepth];
	this->weightsReader.GetBias(outputFeatureDepth, bias);
	convLayerGPU->SetBias(bias);

	if (batchNormalization)
	{
		float * bnScales = new float[outputFeatureDepth];
		float * estimatedMean = new float[outputFeatureDepth];
		float * estimatedVariance = new float[outputFeatureDepth];

		this->weightsReader.GetBatchNormalizationParameters(outputFeatureDepth, bnScales, estimatedMean, estimatedVariance);
		convLayerGPU->SetBatchNormalizationParameters(bnScales, estimatedMean, estimatedVariance);
	}

	this->weightsReader.GetWeights(kernelTensor->GetTensorSize(), convLayerGPU->GetKernelTensorWeights());
	//kernelTensor->InitKernelWeights();
	//kernelTensor->SetKernelWeights(weights);
	
	convLayerGPU->SetPreviousLayer(nullptr);
	convLayerGPU->SetupCUDNN(true);

	this->layers.push_back(convLayerGPU);
}

void CNN::AddConvLayer(TensorDimension kernalTensorDimension, TensorDimension outputTensorDimension, int stride, int padding, int activationType, bool batchNormalization)
{
	if (this->layers.size() == 0)
	{
		return;
	}

	Tensor* inputTensor = this->layers[this->layers.size() - 1]->GetOutputTensor();
	Tensor* kernelTensor = new Tensor(kernalTensorDimension);
	Tensor* outputTensor = new Tensor(outputTensorDimension);
	Conv_Layer_GPU* convLayerGPU = new Conv_Layer_GPU(inputTensor, outputTensor, kernelTensor, stride, padding, activationType, batchNormalization);

	int outputFeatureDepth = outputTensor->GetChannels();
	float * bias = new float[outputFeatureDepth];
	this->weightsReader.GetBias(outputFeatureDepth, bias);
	convLayerGPU->SetBias(bias);

	if (batchNormalization)
	{
		float * bnScales = new float[outputFeatureDepth];
		float * estimatedMean = new float[outputFeatureDepth];
		float * estimatedVariance = new float[outputFeatureDepth];

		this->weightsReader.GetBatchNormalizationParameters(outputFeatureDepth, bnScales, estimatedMean, estimatedVariance);
		convLayerGPU->SetBatchNormalizationParameters(bnScales, estimatedMean, estimatedVariance);
	}

	this->weightsReader.GetWeights(kernelTensor->GetTensorSize(), convLayerGPU->GetKernelTensorWeights());

	/*if (batchNormalization == false)
	{
		float* p = convLayerGPU->GetKernelTensorWeights();
		float t;
		for (size_t i = 1024 - 1; i < 125 * 1024; i++)
		{
			t = p[i];
		}
	}*/
	
	//kernelTensor->InitKernelWeights();
	//kernelTensor->SetKernelWeights(weights);
	
	convLayerGPU->SetPreviousLayer(this->layers[this->layers.size() - 1]);
	convLayerGPU->SetupCUDNN(false);

	this->layers.push_back(convLayerGPU);
}

void CNN::AddMaxPoolLayer(TensorDimension inputTensorDimension, TensorDimension outputTensorDimension, int stride, int padding)
{
}

void CNN::AddMaxPoolLayer(TensorDimension outputTensorDimension, int stride, int padding)
{
	if (this->layers.size() == 0)
	{
		return;
	}

	Tensor* inputTensor = this->layers[this->layers.size() - 1]->GetOutputTensor();
	Tensor* outputTensor = new Tensor(outputTensorDimension);

	MaxPool_Layer_GPU* maxPoolLayerGPU = new MaxPool_Layer_GPU(inputTensor, outputTensor, stride, padding);
	maxPoolLayerGPU->SetPreviousLayer(this->layers[this->layers.size() - 1]);
	maxPoolLayerGPU->SetupCUDNN(false);
	this->layers.push_back(maxPoolLayerGPU);
}





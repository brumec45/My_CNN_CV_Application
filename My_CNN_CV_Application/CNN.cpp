#include "CNN.h"
#include "ImageInput.h"

CNN::CNN()
{
}

void CNN::InitCNN()
{
	Mat testImage = LoadImage("more_of_lenna.jpg");
	Mat myBlob = cv::dnn::blobFromImage(testImage, 1, cv::Size(244, 244), cv::Scalar(0, 0, 0), true, false);
	float * myBlobF = myBlob.ptr<float>();

	TensorDimension td1(1, 3, 244, 244);
	TensorDimension td2(3, 3, 3, 3);
	TensorDimension td3(1, 3, 244, 244);
	AddConvLayer(td1, td2, td3);

	TensorDimension td4(3, 3, 3, 3);
	TensorDimension td5(1, 3, 244, 244);
	AddConvLayer(td4, td5);


	this->layers[0]->SetInputData(myBlobF);

	for (size_t i = 0; i < this->layers.size(); i++)
	{
		this->layers[i]->Forward();
	}
	

	float* outputData = this->layers[this->layers.size() - 1]->GetLayerOutputData();

	Mat result(244, 244, CV_32FC1);
	//float* resultP = result.

	for (int i = 0; i < 244; i++)
	{
		//resultP = result.data + i * result.step;
		for (int j = 0; j < 244; j++)
		{
			//resultP[j] = 100;// h_output[i * 244 + j];
			result.at<float>(i, j) = outputData[i * 244 + j];
		}
	}
	//float a = h_output[10];
	imshow("src", result);
	cv::waitKey(0);
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

	Tensor* inputTensor = this->layers[0]->GetOutputTensor();
	//inputTensor.SetTensorData(myBlobF);

	Tensor* kernelTensor = new Tensor(kernalTensorDimension);
	kernelTensor->InitKernelWeights();

	Tensor* outputTensor = new Tensor(outputTensorDimension);

	Conv_Layer_GPU* convLayerGPU = new Conv_Layer_GPU(inputTensor, outputTensor, kernelTensor);

	convLayerGPU->SetPreviousLayer(this->layers[this->layers.size() - 1]);
	convLayerGPU->SetupCUDNN(false);

	this->layers.push_back(convLayerGPU);
}





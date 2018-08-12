#include "cuda.h"
#include "ImageInput.h"
#include "CNN.h"
#include "RELU_Activation_Kernel.cuh"
#include <chrono>
#include <iostream>
#include "WeightsReader.h"


using namespace std::chrono;

void fileReadTest() {
	FILE * pFile;
	size_t result;
	fopen_s(&pFile, "Weights\\tinyyolov2.weights", "rb");
	if (pFile == NULL) { 
		fputs("File error", stderr); exit(1); 
	}
	float* testp = new float[200];

	result = fread(testp, sizeof(float), 200, pFile);
	if (result != 200) { 
		fputs("Reading error", stderr); exit(3); 
	}
	float t;
	for (size_t i = 134; i < 200; i++)
	{
		t = testp[i];
	}
}

int main(int, char** argv)
{
	//CUDADeviceInfo();
	Mat testImage = LoadImage("more_of_lenna.jpg");
	Mat myBlob = cv::dnn::blobFromImage(testImage, 1, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

	float * myBlobF = myBlob.ptr<float>();
	
	CNN* cnn = new CNN();
	cnn->GenerateTinyYOLOv2Architecture(416, "Weights\\tinyyolov2.weights", true);
	//cnn->GenerateSimpleConvolutionTestArchitecture(416, "Weights\\tinyyolov2.weights", true);
	cnn->SetInput(myBlobF);
	cnn->ForwardPass();
	float* outputData = cnn->GetOutput();
	int outputSize = 13;
	Mat result(outputSize, outputSize, CV_32FC1);
	std::vector<float> testVector;

	float t;
	for (int i = 0; i < outputSize; i++)
	{
		//resultP = result.data + i * result.step;
		for (int j = 0; j < outputSize; j++)
		{
			//resultP[j] = 100;// h_output[i * 244 + j];
			t = outputData[i * outputSize + j];
			testVector.push_back(t);
			result.at<float>(i, j) = outputData[i * outputSize + j];
		}
	}
	
	imshow("src", result);
	cv::waitKey(0);

	VideoCapture cap("D:\\Faks\\Magisterij\\Mag_delo\\Aplikacija\\testVideo.avi");
	int frameCounter = 0;

	time_point<steady_clock> startTime = steady_clock::now();
	time_point<steady_clock> frameTime;

	/*while (cap.isOpened())
	{

		Mat frame;
		cap >> frame;
		Mat myBlob = cv::dnn::blobFromImage(frame, 1, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
		float * myBlobF = myBlob.ptr<float>();

		cnn->SetInput(myBlobF);
		cnn->ForwardPass();
		float* output = cnn->GetOutput();

		imshow("video", frame);
		++frameCounter;
		frameTime = steady_clock::now();

		if (frameTime - startTime >= seconds{ 1 }) {

			std::cout << "fps: " << frameCounter << std::endl;
			frameCounter = 0;
			startTime = frameTime;
		}



		if (waitKey(32) >= 0) break;
	}*/



	system("PAUSE");
	return 0;
}

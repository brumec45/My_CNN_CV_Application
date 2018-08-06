#include "cuda.h"
#include "ImageInput.h"
#include "CNN.h"
#include "RELU_Activation_Kernel.cuh"
#include <chrono>
#include <iostream>
using namespace std::chrono;




int main(int, char** argv)
{
	//CUDADeviceInfo();

	CNN* cnn = new CNN();
	cnn->GenerateTinyYOLOv2Architecture(416);

	VideoCapture cap("D:\\Faks\\Magisterij\\Mag_delo\\Aplikacija\\testVideo.avi");
	int frameCounter = 0;

	time_point<steady_clock> startTime = steady_clock::now();
	time_point<steady_clock> frameTime;

	while (cap.isOpened())
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
	}

	system("PAUSE");
	return 0;
}

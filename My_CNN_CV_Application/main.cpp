#include <omp.h>
#include <opencv2/opencv.hpp>
#include <climits>
#include <stdlib.h> 
#include <vector>
#include "cuda.h"


using namespace std;
using namespace cv;



cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

int main(int, char** argv)
{
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	checkCUDNN(cudnnCreate(&cudnn));
	
	Mat src = load_image("more_of_lenna.jpg");
	//cv::resize(src, src, cv::Size(244, 244));

	//cv::resize(src, image, cv::Size(512, 512));
	//cv::cvtColor(image, image, CV_BGR2YCrCb);
	
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_path]\n");
		return -1;
	}
	Mat myBlob = cv::dnn::blobFromImage(src, 1,cv::Size(244, 244), cv::Scalar(0, 0, 0), true, false);

	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/3,
		/*image_height=*/244,
		/*image_width=*/244));

	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/3,
		/*image_height=*/244,
		/*image_width=*/244));

	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/3,
		/*in_channels=*/3,
		/*kernel_height=*/3,
		/*kernel_width=*/3));

	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/1,
		/*pad_width=*/1,
		/*vertical_stride=*/1,
		/*horizontal_stride=*/1,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			/*memoryLimitInBytes=*/0,
			&convolution_algorithm));

	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes));
	std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
		<< std::endl;

	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);

	int image_bytes = 1 * 3 * 244 * 244 * sizeof(float);

	float* d_input{ nullptr };
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, myBlob.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

	float* d_output{ nullptr };
	cudaMalloc(&d_output, image_bytes);
	cudaMemset(d_output, 0, image_bytes);

	// Mystery kernel
	const float kernel_template[3][3] = {
		{ 1,  1, 1 },
		{ 1, -8, 1 },
		{ 1,  1, 1 }
	};

	float h_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel) {
		for (int channel = 0; channel < 3; ++channel) {
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}

	float* d_kernel{ nullptr };
	cudaMalloc(&d_kernel, sizeof(h_kernel));
	cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(cudnn,
		&alpha,
		input_descriptor,
		d_input,
		kernel_descriptor,
		d_kernel,
		convolution_descriptor,
		convolution_algorithm,
		d_workspace,
		workspace_bytes,
		&beta,
		output_descriptor,
		d_output));

	float* h_output = new float[image_bytes];
	cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

	// Do something with h_output ...



	Mat result(244, 244, CV_32FC1);
	//float* resultP = result.

	for (int i = 0; i < 244; i++)
	{
		//resultP = result.data + i * result.step;
		for (int j = 0; j < 244; j++)
		{
			//resultP[j] = 100;// h_output[i * 244 + j];
			result.at<float>(i, j) = h_output[i * 244 + j];
		}
	}
	//float a = h_output[10];
	imshow("src", result);
	cv::waitKey(0);

	delete[] h_output;
	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);

	cudnnDestroy(cudnn);
	printf(cv::getBuildInformation().c_str());
	system("PAUSE");
	return 0;
}

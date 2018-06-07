#include <omp.h>
#include <opencv2/opencv.hpp>
#include <climits>
#include <stdlib.h> 
#include <vector>

using namespace std;
using namespace cv;

int main(int, char** argv)
{

	Mat src = imread("more_of_lenna.jpg");

	//cv::resize(src, image, cv::Size(512, 512));
	//cv::cvtColor(image, image, CV_BGR2YCrCb);
	
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_path]\n");
		return -1;
	}

	imshow("src", src);
	cv::waitKey(0);
	return 0;
}

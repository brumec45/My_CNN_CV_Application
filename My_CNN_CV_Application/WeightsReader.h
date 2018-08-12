#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <string.h>
/*
FORMAT:
 4 * int
 n * bias (float)
 batch_norm{
	n * scales (float)
	n * means (float)
	n * variances (float)
 }
 n * c * h * w weights (float)
 .
 .
 .

 */
class WeightsReader {

private:


	//std::string weightsFileName;
	//std::ifstream file;
	//ker ifstream nekaj cudno zaokrozuje zaradi nekega razloga :/ 
	FILE * file;
	char* weightsFileName;
	int startOffset = 4;
public:
	WeightsReader();
	WeightsReader(char* weightsFileName);
	void GetWeights(int weightsCount, float * weights);
	void GetBatchNormalizationParameters(int previousConvLayerFeatureDepth, float * biases, float * scales, float * rollingMean, float * rollingVariance);
	void SetFileName(char* weightsFileName);
	void OpenFile(char* weightsFileName);
	void CloseFile();
};

#include "WeightsReader.h"

WeightsReader::WeightsReader()
{
}

WeightsReader::WeightsReader(char* weightsFileName)
{
	this->weightsFileName = weightsFileName;
}



void WeightsReader::GetWeights(int weightsCount, float * weights)
{
	size_t weightsRead;
	weightsRead = fread(weights, sizeof(float), weightsCount, this->file);
	if (weightsRead != weightsCount) {
		fputs("Weights reading error", stderr); exit(3);
	}
	/*float t;
	for (size_t i = 0; i < weightsCount; i++)
	{
		t = weights[i];
	}*/
	/*try
	{
		this->file.read((char*)weights, 72 * sizeof(float));
	}
	catch (std::ifstream::failure e)
	{
		std::string t = e.what();
		char errBuffer[20];
		strerror_s(errBuffer, 20, errno);
		std::cout << "GetWeights Error: " << errBuffer << std::endl;
		std::getchar();
		std::exit(EXIT_FAILURE);
	}*/
				
	/*std::ifstream file(this->weightsFileName);
	std::string line;
	int lineCounter = 0;
	int weightsCounter = 0;

	while (std::getline(file, line))
	{
		if (weightsCounter == weightsCount)
		{
			break;
		}
				
		if (lineCounter >= this->offset)
		{
			float weight;
			std::stringstream ss(line);
			ss >> weight;
			weights[weightsCounter] = weight;
			++weightsCounter;
		}
		++lineCounter;
	}
	file.clear();
	file.seekg(0, std::ios_base::beg);*/
}

void WeightsReader::GetBias(int previousConvLayerFeatureDepth, float * bias)
{
	size_t parametersRead;
	parametersRead = fread(bias, sizeof(float), previousConvLayerFeatureDepth, this->file);
	if (parametersRead != previousConvLayerFeatureDepth) {
		fputs("Bias reading error: ", stderr); exit(3);
	}
}

void WeightsReader::GetBatchNormalizationParameters(int previousConvLayerFeatureDepth, float * bnScales, float * estimatedMean, float * estimatedVariance)
{
	size_t parametersRead;
	/*parametersRead = fread(bnBias, sizeof(float), previousConvLayerFeatureDepth, this->file);
	if (parametersRead != previousConvLayerFeatureDepth) {
		fputs("Batch normalization parameters reading error", stderr); exit(3);
	}*/
	parametersRead = fread(bnScales, sizeof(float), previousConvLayerFeatureDepth, this->file);
	if (parametersRead != previousConvLayerFeatureDepth) {
		fputs("Batch normalization parameters reading error: ", stderr); exit(3);
	}
	parametersRead = fread(estimatedMean, sizeof(float), previousConvLayerFeatureDepth, this->file);
	if (parametersRead != previousConvLayerFeatureDepth) {
		fputs("Batch normalization parameters reading error: ", stderr); exit(3);
	}
	parametersRead = fread(estimatedVariance, sizeof(float), previousConvLayerFeatureDepth, this->file);
	if (parametersRead != previousConvLayerFeatureDepth) {
		fputs("Batch normalization parameters reading error: ", stderr); exit(3);
	}

	/*float t;
	for (size_t i = 0; i < previousConvLayerFeatureDepth; i++)
	{
		t = estimatedVariance[i];
	}*/
	/*try
	{
		
		float t;
		this->file.read((char*)bnBias, 155 * sizeof(float));
		for (size_t i = 135; i < 155; i++)
		{
			t = bnBias[i];
		}

		this->file.read((char*)bnScales, previousConvLayerFeatureDepth * sizeof(float));
		this->file.read((char*)estimatedMean, previousConvLayerFeatureDepth * sizeof(float));
		this->file.read((char*)estimatedVariance, previousConvLayerFeatureDepth * sizeof(float));
	}
	catch (std::ifstream::failure e)
	{
		std::string t = e.what();
		char errBuffer[20];
		strerror_s(errBuffer, 20, errno);
		std::cout << "GetBatchNormalizationParameters Error: " << errBuffer << std::endl;
		std::getchar();
		std::exit(EXIT_FAILURE);
	}*/
	
}

void WeightsReader::SetFileName(char* fileName)
{
	this->weightsFileName = weightsFileName;
}

void WeightsReader::OpenFile(char* fileName)
{
	fopen_s(&file, fileName, "rb");
	if (file == NULL) {
		fputs("File error", stderr); exit(1);
	}

	fseek(file, this->startOffset * sizeof(int), SEEK_SET);
	/*file.exceptions(std::ifstream::failbit);

	try
	{
		this->startOffset = 4;
		this->file.open(fileName, std::ifstream::in);
		//skipaš prve 4 inte
		this->file.seekg(this->startOffset * sizeof(int));
	}
	catch (std::ifstream::failure e)
	{
		char errBuffer[20];
		strerror_s(errBuffer, 20, errno);
		std::cout << "OpenFile Error: " << errBuffer << std::endl;
		std::getchar();
		std::exit(EXIT_FAILURE);
		//std::cerr << "Exception opening file: " << std::strerror(errno) << "\n";
	} */

}

void WeightsReader::CloseFile()
{
	this->startOffset = 4;
	//file.clear();
	fclose(this->file);
}

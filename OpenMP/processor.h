#ifndef PROCESS_H
#define PROCESSOR_H

#include <string>
#include "defines.h"

class processor
{

public:
	processor(const int&, const int&, rgb*);
	void imageConvolution();
	void setGaussianBlurMatrix();
	void RGBToGrayscale();
	void sobelConvolution();
	void nonMaxSuppression();
	void hysteresisThresholding();
	void cannyEdgeDetection();

	// accessor functions
	rgb* getPixels();
	unsigned char* getGrayscaleData();
	int* getGradientMagnitudes();
	double* getEdgeDirections();
	unsigned char* getEdges();

	// destructor
	~processor();

private:
	int height, width;
	int highThresh, lowThresh;
	rgb* pixels;
	unsigned char* grayscaleData;
	int* gradientMagnitudes;
	double* edgeDirections;
	unsigned char* edges;
	double** filtermatrix;

};

#endif
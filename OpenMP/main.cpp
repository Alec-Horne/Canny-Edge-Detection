#include "BitmapImage.h"
#include "defines.h"
#include "processor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>  

// define the window position on screen
int window_x, window_y;

// variables representing the window size
int window_width, window_height;

int main(int argc, char **argv)
{
	const wchar_t* filename = L"jax.bmp";
	BitmapImage* bmi = new BitmapImage(filename);

	// get pixels for convolution
	rgb* r = bmi->getPixels();


	// convolute image
	processor* proc = new processor(bmi->getWidth(), bmi->getHeight(), r);
	proc->cannyEdgeDetection();
	unsigned char* e = proc->getGrayscaleData();

	// set opengl window height and width
	window_width = bmi->getWidth();
	window_height = bmi->getHeight();

	// convert pixel data to Mat for opencv
	cv::Mat image = cv::Mat(window_height, window_width, CV_8UC1, e);

	// create a window and show the image
	cv::namedWindow("Processed Image", 0);
	cv::resizeWindow("Processed Image", window_width, window_height);
	cv::imshow("Processed Image", image);
	
	// center window on screen
	window_x = (1366 - window_width / 2) / 2;
	window_y = (768 - window_height / 2) / 2;
	cv::moveWindow("Processed Image", window_x, window_y);

	// save processed image to file
	cv::imwrite("output.bmp", image);

	// wait for a keystroke in the window before ending
	cv::waitKey(0);
	
	// free memory
	delete[] r;
	delete[] e;
	delete bmi;
	delete proc;
}
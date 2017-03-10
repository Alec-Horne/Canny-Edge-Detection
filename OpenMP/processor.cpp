#include "processor.h"
#include "defines.h"
#include <algorithm>
#include <omp.h>
#include <cmath>
#include "BitmapImage.h"
#include <opencv2/core/cvdef.h>
#include <opencv2/imgproc.hpp>


processor::processor(const int& w, const int& h, rgb* p)
{
	omp_set_num_threads(4);
	width = w;
	height = h;
	pixels = p;
	grayscaleData = new unsigned char[w * h];
	gradientMagnitudes = new int[w * h];
	edgeDirections = new double[w * h];
	edges = new unsigned char[w * h];
	for (int x = 0; x < w * h; x++)
	{
		edges[x] = 0;
	}
}

processor::~processor()
{}

void processor::cannyEdgeDetection()
{
	imageConvolution();
	RGBToGrayscale();
	sobelConvolution();
	nonMaxSuppression();
	hysteresisThresholding();
}

void processor::RGBToGrayscale()
{
	int y, x, grayScale;

	#pragma omp parallel private(y, x, grayScale)
	{
		#pragma omp for schedule(static)
		for (y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++)
			{
				grayScale = static_cast<int>((pixels[y * width + x].r * 0.3) + (pixels[y * width + x].g * 0.59) +
					(pixels[y * width + x].b * 0.11));
				grayscaleData[y * width + x] = grayScale;
			}
		}
	}
}

void processor::imageConvolution()
{

	int fmat[5][5] = {
		2, 4, 5, 4, 2,
		4, 9, 12, 9, 4,
		5, 12, 15, 12, 5,
		4, 9, 12, 9, 4,
		2, 4, 5, 4, 2
	};

	double factor = 1.0 / 159.0;

	int x, y;

	//apply the filter
	#pragma omp parallel private(x, y) 
	{
		#pragma omp for schedule(static)
		for (x = 0; x < width; x++)
		{
			for (y = 0; y < height; y++)
			{
				double red = 0.0, green = 0.0, blue = 0.0;

				//multiply every value of the filter with corresponding image pixel
				for (int filterY = 0; filterY < 5; filterY++)
					for (int filterX = 0; filterX < 5; filterX++)
					{
						int imageX = (x - 5 / 2 + filterX + width) % width;
						int imageY = (y - 5 / 2 + filterY + height) % height;
						red += pixels[imageY * width + imageX].r * fmat[filterY][filterX];
						green += pixels[imageY * width + imageX].g * fmat[filterY][filterX];
						blue += pixels[imageY * width + imageX].b * fmat[filterY][filterX];
					}

				//truncate values smaller than zero and larger than 255
				pixels[y * width + x].r = std::min(std::max(int(factor * red + 1), 0), 255);
				pixels[y * width + x].g = std::min(std::max(int(factor * green + 1), 0), 255);
				pixels[y * width + x].b = std::min(std::max(int(factor * blue + 1), 0), 255);
			}
		}
	}
}

void processor::sobelConvolution()
{
	int fmat_x[3][3] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};
	int fmat_y[3][3] = {
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1
	};

	double G_x, G_y, G;
	int x, y;

	#pragma omp parallel private(x, y, G_x, G_y)
	{
		#pragma omp for schedule(static)
		for (x = 1; x < height - 1; x++)
		{
			for (y = 1; y < width - 1; y++)
			{
				G_x = G_y = 0;
				for (int i = x - 3 / 2; i < x + 3 - 3 / 2; i++) {
					for (int j = y - 3 / 2; j < y + 3 - 3 / 2; j++) {
						G_x += (double)(fmat_x[i - x + 3 / 2][y - j + 3 / 2] * grayscaleData[i * width + j]);
						G_y += (double)(fmat_y[i - x + 3 / 2][y - j + 3 / 2] * grayscaleData[i * width + j]);
					}
				}

				G = sqrt(G_x * G_x + G_y * G_y);
				gradientMagnitudes[x * width + y] = G;
				float angle = atan2(G_y, G_x);

				// if the angle is negative
				if (angle < 0) {
					angle = fmod((angle + 2 * 3.14159), (2 * 3.14159));
				}

				if (angle <= 3.14159 / 8) {
					edgeDirections[x * width + y] = 0;
				}
				else if (angle <= 3 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 45;
				}
				else if (angle <= 5 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 90;
				}
				else if (angle <= 7 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 135;
				}
				else if (angle <= 9 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 0;
				}
				else if (angle <= 11 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 45;
				}
				else if (angle <= 13 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 90;
				}
				else if (angle <= 15 * 3.14159 / 8) {
					edgeDirections[x * width + y] = 135;
				}
				else { 
					edgeDirections[x * width + y] = 0;
				}
			}
		}
	}
}

void processor::nonMaxSuppression()
{
	int row, col;

	// iterate over the height of the photo matrix
	#pragma omp parallel private(row, col) 
	{
	#pragma omp for schedule(static)
		for (row = 1; row < height - 1; row++) {
			// iterate over the columns of the photo matrix
			for (col = 1; col < width - 1; col++) {
				// These variables are used to address the matrices more easily
				const size_t POS = row * width + col;
				const size_t N = (row - 1) * width + col;
				const size_t NE = (row - 1) * width + (col + 1);
				const size_t E = row * width + (col + 1);
				const size_t SE = (row + 1) * width + (col + 1);
				const size_t S = (row + 1) * width + col;
				const size_t SW = (row + 1) * width + (col - 1);
				const size_t W = row * width + (col - 1);
				const size_t NW = (row - 1) * width + (col - 1);
				
				int val = gradientMagnitudes[POS];
				if (val > 255)
					val = 255;

				switch ((int)edgeDirections[POS]) {
				case 0:
					// supress me if my neighbor has larger magnitude
					if (gradientMagnitudes[POS] <= gradientMagnitudes[E] ||
						gradientMagnitudes[POS] <= gradientMagnitudes[W]) {
						edges[POS] = 0;
					}
					// otherwise, copy my value to the output buffer
					else {
						edges[POS] = val;
					}
					break;

				case 45:
					// supress me if my neighbor has larger magnitude
					if (gradientMagnitudes[POS] <= gradientMagnitudes[NE] ||
						gradientMagnitudes[POS] <= gradientMagnitudes[SW]) {
						edges[POS] = 0;
					}
					// otherwise, copy my value to the output buffer
					else {
						edges[POS] = val;
					}
					break;

				case 90:
					// supress me if my neighbor has larger magnitude
					if (gradientMagnitudes[POS] <= gradientMagnitudes[N] ||
						gradientMagnitudes[POS] <= gradientMagnitudes[S]) {
						edges[POS] = 0;
					}
					// otherwise, copy my value to the output buffer
					else {
						edges[POS] = val;
					}
					break;

				case 135:
					// supress me if my neighbor has larger magnitude
					if (gradientMagnitudes[POS] <= gradientMagnitudes[NW] ||
						gradientMagnitudes[POS] <= gradientMagnitudes[SE]) {
						edges[POS] = 0;
					}
					// otherwise, copy my value to the output buffer
					else {
						edges[POS] = val;
					}
					break;

				default:
					edges[POS] = val;
					break;
				}
			}
		}
	}
}


void processor::hysteresisThresholding()
{
	int* temp = new int[width * height];
	for (int x = 0; x < width * height; x++)
	{
		grayscaleData[x] = 0;
		temp[x] = 0;
	}

	int tmax = 70;
	int tmin = 35;
	int j, i;
	#pragma omp parallel private(j, i)
	{
		#pragma omp for schedule(static)
		for (j = 1; j < height - 1; j++) {
			for (i = 1; i < width - 1; i++) {
				if (edges[j * i] >= tmax && grayscaleData[j * i] == 0) {
					grayscaleData[j * i] = 255;
					int nedges = 1;
					temp[0] = j * i;

					do {
						nedges--;
						const int t = temp[nedges];

						int nbs[8];
						nbs[0] = t - width;
						nbs[1] = t + width; 
						nbs[2] = t + 1; 
						nbs[3] = t - 1; 
						nbs[4] = nbs[0] + 1;
						nbs[5] = nbs[0] - 1;
						nbs[6] = nbs[1] + 1;
						nbs[7] = nbs[1] - 1; 

						for (int k = 0; k < 8; k++)
							if (edges[nbs[k]] >= tmin && grayscaleData[nbs[k]] == 0) {
								grayscaleData[nbs[k]] = 255;
								temp[nedges] = nbs[k];
								nedges++;
							}
					} while (nedges > 0);
				}
			}
		}
	}
}

void processor::setGaussianBlurMatrix()
{
	//filtermatrix
}

rgb* processor::getPixels()
{
	return pixels;
}

double* processor::getEdgeDirections()
{
	return edgeDirections;
}

int* processor::getGradientMagnitudes()
{
	return gradientMagnitudes;
}

unsigned char* processor::getGrayscaleData()
{
	return grayscaleData;
}

unsigned char* processor::getEdges()
{
	return edges;
}
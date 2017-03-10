#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "cpu_bitmap.h"
#include "bitmap_help.h"
#include <algorithm>
#include <Windows.h>

__global__ void kernel_grayscale(unsigned char*, unsigned char*, int, int);
__global__ void kernel_gaussian_blur(unsigned char*, unsigned char*, int, int);
__global__ void kernel_sobel_filter(unsigned char*, unsigned char*, unsigned char*, int, int);
__global__ void kernel_non_max_suppression(unsigned char*, unsigned char*, unsigned char*, int, int);
__global__ void kernel_hysteresis_thresholding(unsigned char*, unsigned char*, int, int);
__device__ int device_min(int, int);
__device__ int device_max(int, int);
__host__ void imgProc(unsigned char*, int, int, int);


__host__ void imgProc(unsigned char *map, int size, int width, int height) {
   
	/* Variables to time CUDA execution */
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
    /* Allocate device memory for the result. */
    unsigned char* device_input_data;
    unsigned char* device_input_data2;
    unsigned char* device_output_data;
    unsigned char* device_edgedirection_output;
	unsigned char* host_output = new unsigned char[width * height];
		
    cudaError_t err1 = cudaMalloc((void**)&device_input_data, size);
	cudaError_t err2 = cudaMalloc((void**)&device_input_data2, size / 4);
    cudaError_t err3 = cudaMalloc((void**)&device_output_data, size / 4);
	cudaError_t err4 = cudaMalloc((void**)&device_edgedirection_output, size / 4);

	if (err1 != cudaSuccess) {
		printf("%s", cudaGetErrorString(err1));
		exit(EXIT_FAILURE);
	}
	if (err2 != cudaSuccess) {
		printf("%s", cudaGetErrorString(err2));
		exit(EXIT_FAILURE);
	}
	if (err3 != cudaSuccess) {
		printf("%s", cudaGetErrorString(err3));
		exit(EXIT_FAILURE);
	}
	if (err4 != cudaSuccess) {
		printf("%s", cudaGetErrorString(err4));
		exit(EXIT_FAILURE);
	}
	
    /* Copy the input data to the device. */
    cudaMemcpy(device_input_data, map, size, cudaMemcpyHostToDevice);
	
    /* Launch the kernel! */
    dim3 grid(64, 64, 1);
    dim3 block(width / 64 + 1, height / 64 + 1, 1);

	kernel_grayscale<<<grid, block>>>(device_input_data, device_output_data, height, width);
	cudaMemcpy(device_input_data2, device_output_data, size / 4, cudaMemcpyDeviceToDevice);
	
	kernel_gaussian_blur<<<grid, block>>>(device_input_data2, device_output_data, height, width);
	cudaMemcpy(device_input_data2, device_output_data, size / 4, cudaMemcpyDeviceToDevice);
	
	kernel_sobel_filter<<<grid, block>>>(device_input_data2, device_output_data, device_edgedirection_output, height, width);
	cudaMemcpy(device_input_data2, device_output_data, size / 4, cudaMemcpyDeviceToDevice);
	
	kernel_non_max_suppression<<<grid, block>>>(device_input_data2, device_output_data, device_edgedirection_output, height, width);
	cudaMemcpy(device_input_data2, device_output_data, size / 4, cudaMemcpyDeviceToDevice);
	
	kernel_hysteresis_thresholding<<<grid, block>>>(device_input_data2, device_output_data, height, width);
	cudaMemcpy(host_output, device_output_data, size / 4, cudaMemcpyDeviceToHost);
	
	int count = 0;
	for (int x = 0; x < width * height; x++) {
		map[count] = host_output[x];
		map[count + 1] = host_output[x];
		map[count + 2] = host_output[x];
		map[count + 3] = 0;
		count += 4;
	}
	
    cudaFree(device_input_data);
    cudaFree(device_input_data2);
	cudaFree(device_output_data);
	cudaFree(device_edgedirection_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Total execution time: %f ms\n", time);
}


__global__ void kernel_grayscale(unsigned char* device_input_data, unsigned char* device_output_data, int height, int width) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Bound check */
    if (x < 0 || x > width || y > height || y < 0)
        return;
	
	int grayOffset = y * width + x;
	int rgbOffset = grayOffset * 4; 
	unsigned char r = device_input_data[rgbOffset];
	unsigned char g = device_input_data[rgbOffset + 1];
	unsigned char b = device_input_data[rgbOffset + 2];
	
	int grayscale = 0.21f * r + 0.71f * g + 0.07f * b;
 
    if (grayscale < 0)
        grayscale = 0;
    if (grayscale > 255)
        grayscale = 255;

    device_output_data[grayOffset] = grayscale;
}


__global__ void kernel_gaussian_blur(unsigned char* device_input_data2, unsigned char* device_output_data, int height, int width) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Bound check */
    if (x < 0 || x > width || y > height || y < 0)
        return;
	
	
	int fmat[5][5] = {
		2, 4, 5, 4, 2,
		4, 9, 12, 9, 4,
		5, 12, 15, 12, 5,
		4, 9, 12, 9, 4,
		2, 4, 5, 4, 2
	};

	double factor = 1.0 / 159.0;
	double val = 0.0;

	/* Multiply every value of the filter with corresponding image pixel */
	for (int filterY = 0; filterY < 5; filterY++)
		for (int filterX = 0; filterX < 5; filterX++)
		{
			int imageX = (x - 5 / 2 + filterX + width) % width;
			int imageY = (y - 5 / 2 + filterY + height) % height;
			val += device_input_data2[imageY * width + imageX] * fmat[filterY][filterX];
		}

	/* Truncate to 0 or 255
	device_output_data[y * width + x] = device_min(device_max(int(factor * val + 1), 0), 255);
}


__global__ void kernel_sobel_filter(unsigned char* device_input_data2, unsigned char* device_output_data, unsigned char* device_edgedirection_output, int height, int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Bound check */
    if (x < 1 || x > width - 1 || y > height - 1 || y < 1)
        return;
		
    /* To detect horizontal lines, G_x. */
    const int fmat_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    /* To detect vertical lines, G_y */
    const int fmat_y[3][3]  = {
        {-1, -2, -1},
        {0,   0,  0},
        {1,   2,  1}
    };

    double G_x = 0;
	double G_y = 0;
	int G;

	for (int i = y - 3 / 2; i < y + 3 - 3 / 2; i++) {
		for (int j = x - 3 / 2; j < x + 3 - 3 / 2; j++) {
			G_x += (double)(fmat_x[i - y + 3 / 2][x - j + 3 / 2] * device_input_data2[i * width + j]);
			G_y += (double)(fmat_y[i - y + 3 / 2][x - j + 3 / 2] * device_input_data2[i * width + j]);
		}
	}

	/* Magnitude */
	G = sqrt(G_x * G_x + G_y * G_y);
    
    if (G < 0)
        G = 0;
    if (G > 255)
        G = 255;

    device_output_data[y * width + x] = G;
	
	float angle = atan2(G_y, G_x);

	// if negative, add 2*pi mod 2*pi for value
	if (angle < 0) {
		angle = fmod((angle + 2 * 3.14159), (2 * 3.14159));
	}

	if (angle <= 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 0;
	}
	else if (angle <= 3 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 45;
	}
	else if (angle <= 5 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 90;
	}
	else if (angle <= 7 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 135;
	}
	else if (angle <= 9 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 0;
	}
	else if (angle <= 11 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 45;
	}
	else if (angle <= 13 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 90;
	}
	else if (angle <= 15 * 3.14159 / 8) {
		device_edgedirection_output[y * width + x] = 135;
	}
	else {
		device_edgedirection_output[y * width + x] = 0;
	}
}

__global__ void kernel_non_max_suppression(unsigned char* device_input_data2, unsigned char* device_output_data, unsigned char* device_edgedirection_output, int height, int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Bound check */
    if (x < 1 || x > width - 1 || y > height - 1 || y < 1)
        return;
		
	int POS = y * width + x;
	int N = (y - 1) * width + x;
	int NE = (y - 1) * width + (x + 1);
	int E = y * width + (x + 1);
	int SE = (y + 1) * width + (x + 1);
	int S = (y + 1) * width + x;
	int SW = (y + 1) * width + (x - 1);
	int W = y * width + (x - 1);
	int NW = (y - 1) * width + (x - 1);
	
	int val = device_input_data2[POS];
					
	switch ((int)device_edgedirection_output[y * width + x]) {
		case 0:
			if (device_input_data2[POS] <= device_input_data2[E] || device_input_data2[POS] <= device_input_data2[W]) {
				device_output_data[POS] = 0;
			}
			else {
				device_output_data[POS] = val;
			}
			break;

		case 45:
			if (device_input_data2[POS] <= device_input_data2[NE] || device_input_data2[POS] <= device_input_data2[SW]) {
				device_output_data[POS] = 0;
			}
			else {
				device_output_data[POS] = val;
			}
			break;

		case 90:
			if (device_input_data2[POS] <= device_input_data2[N] || device_input_data2[POS] <= device_input_data2[S]) {
				device_output_data[POS] = 0;
			}
			else {
				device_output_data[POS] = val;
			}
			break;
=
		case 135:
			if (device_input_data2[POS] <= device_input_data2[NW] || device_input_data2[POS] <= device_input_data2[SE]) {
				device_output_data[POS] = 0;
			}
			else {
				device_output_data[POS] = val;
			}
			break;

		default:
			device_output_data[POS] = val;
			break;
	}
}


__global__ void kernel_hysteresis_thresholding(unsigned char* device_input_data2, unsigned char* device_output_data, int height, int width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Bound check */
    if (x < 0 || x > width || y > height || y < 0)
        return;

	float lowThresh = 35;
	float highThresh = 70;

	// These variables are offset by one to avoid seg. fault errors
    // As such, this kernel ignores the outside ring of pixels
	int pos = y * width + x;

    unsigned char magnitude = device_input_data2[pos];
    
    if (magnitude >= highThresh)
        device_output_data[pos] = 255;
    else if (magnitude <= lowThresh)
        device_output_data[pos] = 0;
    else
    {
        float med = (highThresh + lowThresh) / 2;
        
        if (magnitude >= med)
            device_output_data[pos] = 255;
        else
            device_output_data[pos] = 0;
    }
}


__device__ int device_min(int a, int b) {
	return (a < b) ? a : b;
}

__device__ int device_max(int a, int b) {
	return (a < b) ? b : a;
}


int main(void) {
   char fname[50];
   FILE* infile;
   unsigned short ftype;
   tagBMFH bitHead;
   tagBMIH bitInfoHead;
   tagRGBQ *pRgb;

   printf("Please enter the .bmp file name: ");
   scanf("%s", fname);
   strcat(fname,".bmp");
   infile = fopen(fname, "rb");

   if (infile != NULL) {
      printf("File open successful.\n");
      fread(&ftype, 1, sizeof(unsigned short), infile);
      if (ftype != 0x4d42)
      {
         printf("File not .bmp format.\n");
         return 1;
      }
      fread(&bitHead, 1, sizeof(tagBMFH), infile);
      fread(&bitInfoHead, 1, sizeof(tagBMIH), infile);      
   }
   else {
      printf("File open fail.\n");
      return 1;
   }

   if (bitInfoHead.biBitCount < 24) {
      long nPlateNum = long(pow(2, double(bitInfoHead.biBitCount)));
      pRgb = (tagRGBQ *)malloc(nPlateNum * sizeof(tagRGBQ));
      memset(pRgb, 0, nPlateNum * sizeof(tagRGBQ));
      int num = fread(pRgb, 4, nPlateNum, infile);
   }

   int width = bitInfoHead.biWidth;
   int height = bitInfoHead.biHeight;
   int l_width = 4 * ((width * bitInfoHead.biBitCount + 31) / 32);
   long nData = height * l_width;
   unsigned char *pColorData = (unsigned char *)malloc(nData);
   memset(pColorData, 0, nData);
   fread(pColorData, 1, nData, infile);

   fclose(infile);
   
   CPUBitmap dataOfBmp(width, height);
   unsigned char *map = dataOfBmp.get_ptr();

   if (bitInfoHead.biBitCount < 24) {
      int k, index = 0;
      if (bitInfoHead.biBitCount == 1) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 8;
               mixIndex = pColorData[k];
               if (j % 8 < 7) mixIndex = mixIndex << (7 - (j % 8));
               mixIndex = mixIndex >> 7;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 2) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 4;
               mixIndex = pColorData[k];
               if (j % 4 < 3) mixIndex = mixIndex << (6 - 2 * (j % 4));
               mixIndex = mixIndex >> 6;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 4) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j / 2;
               mixIndex = pColorData[k];
               if (j % 2 == 0) mixIndex = mixIndex << 4;
               mixIndex = mixIndex >> 4;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 8) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j;
               mixIndex = pColorData[k];
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
       else if (bitInfoHead.biBitCount == 16) {
         for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
               unsigned char mixIndex = 0;
               k = i * l_width + j * 2;
               unsigned char shortTemp = pColorData[k + 1] << 8;
               mixIndex = pColorData[k] + shortTemp;
               map[index * 4 + 0] = pRgb[mixIndex].rgbRed;
               map[index * 4 + 1] = pRgb[mixIndex].rgbGreen;
               map[index * 4 + 2] = pRgb[mixIndex].rgbBlue;
               map[index * 4 + 3] = pRgb[mixIndex].rgbReserved;
               index++;
            }
       }
   }
   else {
      int k, index = 0;
      for (int i = 0; i < height; i++)
         for (int j = 0; j < width; j++) {
            k = i * l_width + j * 3;
            map[index * 4 + 0] = pColorData[k + 2];
            map[index * 4 + 1] = pColorData[k + 1];
            map[index * 4 + 2] = pColorData[k];
            index++;
         }
   }
   
   imgProc(map, dataOfBmp.image_size(), width, height);
   dataOfBmp.display_and_exit();
   return 0;
}
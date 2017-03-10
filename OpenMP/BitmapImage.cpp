#include "BitmapImage.h"
#include "defines.h"
#include <Unknwn.h>    
#include <windows.h>
#include <gdiplus.h>
#include <vector>
#include <iostream>
#include <sstream>
#include "processor.h"
#pragma comment(lib,"gdiplus.lib")


BitmapImage::BitmapImage(const wchar_t* fn)
{
	width = 0;
	height = 0;
	filename = fn;
	pixels = getPixels(filename);
	stride = 0;
}

BitmapImage::~BitmapImage() {}

rgb* BitmapImage::getPixels(const wchar_t* name)
{
	// initialize windows gdi+
	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);

	// read bitmap file and get width and height
	Gdiplus::Bitmap* bitmap = new Gdiplus::Bitmap(name);
	width = bitmap->GetWidth();
	height = bitmap->GetHeight();

	// declare bitmapdata and rect for storing bit data from bmp file
	Gdiplus::BitmapData* bitmapData = new Gdiplus::BitmapData;
	Gdiplus::Rect rect(0, 0, width, height);

	// lock bits into memory
	bitmap->LockBits(&rect, Gdiplus::ImageLockModeRead | Gdiplus::ImageLockModeWrite, PixelFormat24bppRGB, bitmapData);
	unsigned char* p = static_cast<unsigned char*>(bitmapData->Scan0);
	stride = bitmapData->Stride;

	// get rgb values for return 
	rgb* ret = charToRGB(width, height, p);

	// unlock bits from memory
	bitmap->UnlockBits(bitmapData);

	// shutdown windows gdi+
	delete bitmap;
	delete bitmapData;

	Gdiplus::GdiplusShutdown(gdiplusToken);

	// return array containing pixel values
	return ret;
}


rgb* BitmapImage::charToRGB(int& w, int& h, unsigned char* pixels)
{
	rgb* ret = new rgb[w * h];
	int count = 0;

	for (int x = 0; x < w * h; x++)
	{
		ret[x].b = pixels[count++];
		ret[x].g = pixels[count++];
		ret[x].r = pixels[count++];
	}

	return ret;
}


unsigned char* BitmapImage::RGBToChar(const int& w, const int& h, rgb* p)
{
	unsigned char* ret = new unsigned char[w * h * 3];
	int count = 0;

	for (int x = 0; x < w * h; x++)
	{
		ret[count++] = p[x].b;
		ret[count++] = p[x].g;
		ret[count++] = p[x].r;
	}

	return ret;
}

int BitmapImage::getWidth()
{
	return this->width;
}

int BitmapImage::getHeight()
{
	return this->height;
}

int BitmapImage::getStride()
{
	return this->stride;
}

rgb* BitmapImage::getPixels() const
{
	return this->pixels;
}
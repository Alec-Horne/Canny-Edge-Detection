#ifndef BITMAPIMAGE_H
#define BITMAPIMAGE_H

#include "defines.h"
#include <windows.h>

class BitmapImage
{

public:
	BitmapImage(const wchar_t*);
	~BitmapImage();
	rgb* charToRGB(int&, int&, unsigned char*);
	unsigned char* RGBToChar(const int&, const int&, rgb*);
	
	// accessors
	int getWidth();
	int getHeight();
	int getStride();
	rgb* getPixels() const;

private:
	int width;
	int height;
	const wchar_t* filename;
	int stride;
	rgb* pixels;
	rgb* getPixels(const wchar_t*);
};

#endif
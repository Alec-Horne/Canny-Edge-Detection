typedef struct tagBMFH {
   unsigned long bfSize;
   unsigned short bfRes1;
   unsigned short bfRes2;
   unsigned long bfOffBits;
} BMFH;

typedef struct tagBMIH {
   unsigned long biSize;
   long biWidth;
   long biHeight;
   unsigned short biPlanes;
   unsigned short biBitCount;
   unsigned long biCompression;
   unsigned long biSizeImage;
   long biXPelsPerMeter;
   long biYPelsPerMeter;
   unsigned long biClrUsed;
   unsigned long biClrImportant;
} BMIH;

typedef struct tagRGBQ {
   unsigned char rgbBlue;
   unsigned char rgbGreen;
   unsigned char rgbRed;
   unsigned char rgbReserved;
} RGBQ;

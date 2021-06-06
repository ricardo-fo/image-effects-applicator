#ifndef PPMIMAGE_H
#define PPMIMAGE_H

#include <iostream>
using namespace std;

typedef struct {
  unsigned char r, g, b;
} PPMPixel;

class PPMImage {
  public:
    PPMImage();
    ~PPMImage();
    // friend istream& operator >>(ifstream& inputStream, PPMImage& img);
    // friend ostream& operator <<(ofstream& outputStream, const PPMImage& img);
    string magicNumber; // A "magic number" for identifying the file type
    int width; // Width of the image
    int height; // Height of the image
    int maxColorVal; // Maximum color value
    PPMPixel * pixel;
    char * threeChan; // A series of rows and columns (raster) that stores important binary image data
};

#endif

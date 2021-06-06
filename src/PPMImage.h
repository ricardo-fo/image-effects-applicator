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

    int width; // Width of the image
    int height; // Height of the image
    int maxColorVal; // Maximum color value
    PPMPixel * pixel;
};

#endif

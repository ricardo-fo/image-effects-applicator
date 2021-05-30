#ifndef PPMIMAGE_H
#define PPMIMAGE_H

#include <iostream>
using namespace std;

class PPMImage {
  public:
    PPMImage(){;};
    ~PPMImage(){
      delete threeChan;
    }
    friend istream& operator >>(ifstream& inputStream, PPMImage& img);
    friend ostream& operator <<(ofstream& outputStream, const PPMImage& img);
    void grayscale();
    void censored();
    void sepia();

  private:
    string magicNumber; // A "magic number" for identifying the file type
    int width; // Width of the image
    int height; // Height of the image
    int maxColorVal; // Maximum color value
    char *threeChan; // A series of rows and columns (raster) that stores important binary image data
};

#endif

#include <iostream>
#include "CImg.h"

using namespace std;

int main()
{
    cimg_library::CImg<unsigned char> image("/home/ricardo/Pictures/Wallpapers/pink_floyd.jpg");
    image.save("result.ppm");
    return 0;
}

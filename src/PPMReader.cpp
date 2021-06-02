#include "PPMReader.h"
#include "PPMImage.h"
#include <cstring>
#include <iostream>
#include <ios>

using namespace std;

PPMReader::PPMReader(const char * _path) {
  path = new char[strlen(_path) + 1];
  strcpy(path, _path);
}

void PPMReader::load() {
  FILE * fr = fopen(path, "r");

}

PPMImage& PPMReader::getImage() {
  return image;
}

void PPMReader::setPath(const char * _path) {
  path = new char[strlen(_path) + 1];
  strcpy(path, _path);
}

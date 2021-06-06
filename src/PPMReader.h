#ifndef PPMREADER_H
#define PPMREADER_H

#include "PPMImage.h"
#include <cstring>
#include <iostream>

class PPMReader {
  private:
    char * path;
    PPMImage * image;

  public:
    PPMReader(const char *);

    // Carrega a imagem ppm
    void load();

    // Salva uma imagem ppm
    void write(const char *);

    // // Retorna um objeto contendo os dados da imagem PPM
    PPMImage * getImage();

    // Muda o caminho até o arquivo ppm
    void setPath(const char *);
};

#endif //PPMREADER_H

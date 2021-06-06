#include "PPMReader.h"
#include "PPMImage.h"
#include <cstring>
#include <iostream>
// #include <ios>

using namespace std;

PPMReader::PPMReader(const char * _path) {
  path = new char[strlen(_path) + 1];
  strcpy(path, _path);
}

void PPMReader::load() {
  char pSix[10];

  // Abre a imagem
  FILE * stream = fopen(path, "rb");
  if (stream == NULL) {
    cout << "Erro ao abrir o arquivo " << path << endl;
    exit(1);
  }

  // Verifica se o arquivo é do formato PPM
  if (fscanf(stream, "%s", pSix) <= 0 || strncmp(pSix, "P6", 10) != 0) {
    cout << "Erro ao ler o arquivo PPM. Arquivo: " << path << endl;
    exit(1);
  }

  PPMImage * img;
  img = (PPMImage *) malloc(sizeof(PPMImage));
  if (!img) {
    cout << "Erro ao alocar memória." << endl;
    exit(1);
  }

  if (fscanf(stream, "%d %d", &(img->width), &(img->height)) != 2) {
    cout << "Tamanho da imagem inválido." << endl;
    exit(1);
  }

  if (fscanf(stream, "%d", &(img->maxColorVal)) != 1) {
    cout << "A imagem contém um range de RGB inválido." << endl;
    exit(1);
  }

  while(fgetc(stream) != '\n');
  cout << "X: " << img->width << endl;
  cout << "Y: " << img->height << endl;

  img->pixel = (PPMPixel *) malloc(sizeof(PPMPixel) * img->width * img->height);
  if (!img) {
    cout << "Erro ao alocar memória." << endl;
    exit(1);
  }

  if ((int)fread(img->pixel, 3 * img->width, img->height, stream) != img->height) {
    cout << "Erro ao carregar a imagem." << endl;
    exit(1);
  }

  fclose(stream);

  image = img;
}

void PPMReader::write(const char * filename) {
  FILE * stream = fopen(filename, "wb");
  if (stream == NULL) {
    cout << "Erro ao abrir o arquivo: " << path << endl;
    exit(1);
  }

  fprintf(stream, "P6\n");
  fprintf(stream, "%d %d\n", image->width, image->height);
  fprintf(stream, "%d\n", image->maxColorVal);
  fwrite(image->pixel, 3 * image->width, image->height, stream);
  fclose(stream);
}

PPMImage * PPMReader::getImage() {
  return image;
}

void PPMReader::setPath(const char * _path) {
  path = new char[strlen(_path) + 1];
  strcpy(path, _path);
}

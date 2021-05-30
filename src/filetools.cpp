#include <iostream>
#include "filetools.h"
#include "utils.h"
#include "CImg.h"

using namespace std;

/**
 * Function: has_file
 * -------------------------------------
 * Verifica se um arquivo existe.
 *
 * param: const char * path - Caminho até o arquivo.
 *
 * returns: inline bool
 */
bool has_file (const char * path) {
  struct stat buffer;
  return (stat (path, &buffer) == 0);
}

/**
 * Function: to_ppm
 * ---------------------------------
 * Gera a imagem .ppm a partir de um caminho absoluto até a imagem original.
 *
 * param: const char * path - Caminho absoluto até o arquivo.
 *
 * returns: void
 */
char * to_ppm(const char * path) {
  // Gera o nome do arquivo
  char * filename = get_filename(path);
  filename = strcat(filename, ".ppm");

  // Verifica se o nome do arquivo já existe
  while (has_file(filename)) {
    filename = get_filename(path);
    string random = to_string(rand() % 1000);
    strcat(filename, read_str(random));
    strcat(filename, ".ppm");
  }

  // Gera a imagem .ppm
  cimg_library::CImg<unsigned char> image(path);
  image.save(filename);

  return filename;
}

/**
 * Function: get_filename
 * ---------------------------------------------
 * Extraí o nome de um arquivo, sem a sua extensão, de um caminho absoluto.
 *
 * param: const char * path - Caminho absoluto até o arquivo.
 *
 * returns: char *
 */
char * get_filename(const char * path) {
  string cpp_path = string(path);
  size_t last_bar = cpp_path.find_last_of("/");
  size_t last_dot = cpp_path.find_last_of(".");
  string filename = cpp_path.substr(last_bar + sizeof(char), last_dot - last_bar - 1);

  return read_str(filename);
}

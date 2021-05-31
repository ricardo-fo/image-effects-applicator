#include <iostream>
#include <climits>
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
  char * fullpath = new char[strlen(filename) + sizeof(char) * 5];
  strcpy(fullpath, "img/");
  strcat(fullpath, filename);
  char * c_random;
  string random;

  // Verifica se o nome do arquivo já existe
  while (has_file(fullpath)) {
    // Gera o novo nome
    filename = get_filename(path);
    random = to_string(rand() % 1000);
    c_random = read_str(random);
    strcat(filename, c_random);
    strcat(filename, ".ppm");
    free(c_random);

    // Cria o caminho completo
    fullpath = new char[strlen(filename) + sizeof(char) * 5];
    strcpy(fullpath, "img/");
    strcat(fullpath, filename);
  }

  // Gera a imagem .ppm
  cimg_library::CImg<unsigned char> image(path);
  image.save(fullpath);

  return fullpath;
}

/**
 * Function: get_filename
 * ---------------------------------------------
 * Extrai o nome de um arquivo, sem a sua extensão, de um caminho absoluto.
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

/**
 * Function: extract_effects
 * --------------------------------------------------
 * Extrai os efeitos de uma string, criando um vetor de efeitos.
 *
 * param: const char * effects - String com os efeitos informados pelo usuário.
 *
 * returns: int *
 */
int * extract_effects(const char * effects) {
  const char * delim = ",";
  char * copy = new char[strlen(effects) + 1];    // Cópia da string effects para ser manipulada
  char * token = new char[strlen(effects) + 1];   // Token, i.e. efeitos
  char * sanitezed = new char[strlen(token) + 1]; // Token sanitizado
  int * effects_arr;
  size_t size;

  // Extrai os efeitos da string copy
  strcpy(copy, effects);
  token = strtok(copy, delim);
  while(token != NULL) {
    strcpy(sanitezed, token);

    // Realoca o tamanho do vetor e insere o novo elemento
    size += sizeof(int);
    effects_arr = (int *) realloc(effects_arr, size);
    effects_arr[(int)(size / sizeof(int)) - 1] = atoi(trim(sanitezed));

    // Busca pelo próximo token
    token = strtok(NULL, delim);
  }

  // Realoca o vetor para indicar seu fim
  size += sizeof(int);
  effects_arr = (int *) realloc(effects_arr, size);
  effects_arr[(int)(size / sizeof(int)) - 1] = INT_MIN;

  return effects_arr;
}

/**
 * Function: ltrim
 * ------------------------------
 * Lê altera uma string removendo seus espaços à esquerda.
 *
 * param: char * str - String a ser alterada.
 *
 * returns: char *
 */
char * ltrim(char * str) {
  while(isspace(*str)) str++;
  return str;
}


/**
 * Function: rtrim
 * ------------------------------
 * Lê altera uma string removendo seus espaços à direita.
 *
 * param: char * str - String a ser alterada.
 *
 * returns: char *
 */
char * rtrim(char * str) {
  char * back = str + strlen(str);
  while(isspace(*--back));
  *(back + 1) = '\0';
  return str;
}

/**
 * Function: trim
 * ------------------------------
 * Lê altera uma string removendo seus espaços.
 *
 * param: char * str - String a ser alterada.
 *
 * returns: char *
 */
char * trim(char * str) {
  return rtrim(ltrim(str));
}

/**
 * Function: apply_effects
 * ----------------------------------------------
 * Lê o caminho até o arquivo .ppm, espera que o usuário informe
 *
 * param: const char * path - Caminho até o arquivo .ppm
 *
 * returns: void
 */
void apply_effects(const char * path, const int * effects) {
  cout << path << endl;
  // cout << effects << endl;
}

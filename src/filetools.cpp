#include <iostream>
#include <climits>
#include <omp.h>
#include "filetools.h"
#include "utils.h"
#include "CImg.h"
#include "PPMReader.h"
#include "PPMImage.h"

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
bool has_file(const char *path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
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
char *to_ppm(const char *path)
{
  // Gera o nome do arquivo
  char *filename = get_filename(path);
  filename = strcat(filename, ".ppm");
  char *fullpath = new char[strlen(filename) + sizeof(char) * 5];
  strcpy(fullpath, "img/");
  strcat(fullpath, filename);
  char *c_random;
  string random;

  // Verifica se o nome do arquivo já existe
  while (has_file(fullpath))
  {
    // Gera o novo nome
    filename = get_filename(path);
    random = to_string(rand() % 1000);
    c_random = read_str(random);
    strcat(filename, c_random);
    strcat(filename, ".ppm");

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
char *get_filename(const char *path)
{
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
int *extract_effects(const char *effects)
{
  const char *delim = ",";
  char *copy = new char[strlen(effects) + 1];    // Cópia da string effects para ser manipulada
  char *token = new char[strlen(effects) + 1];   // Token, i.e. efeitos
  char *sanitized = new char[strlen(token) + 1]; // Token sanitizado
  int *effects_arr = (int *)malloc(0);           // Reserva espaço na memória para o vetor de efeitos
  size_t size = 0;

  // Extrai os efeitos da string copy
  strcpy(copy, effects);
  token = strtok(copy, delim);
  while (token != NULL)
  {
    strcpy(sanitized, token);

    // Realoca o tamanho do vetor e insere o novo elemento
    size += sizeof(int);
    effects_arr = (int *)realloc(effects_arr, size);
    effects_arr[(int)(size / sizeof(int)) - 1] = atoi(trim(sanitized));

    // Busca pelo próximo token
    token = strtok(NULL, delim);
  }

  // Realoca o vetor para indicar seu fim
  size += sizeof(int);
  effects_arr = (int *)realloc(effects_arr, size);
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
char *ltrim(char *str)
{
  while (isspace(*str))
    str++;
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
char *rtrim(char *str)
{
  char *back = str + strlen(str);
  while (isspace(*--back))
    ;
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
char *trim(char *str)
{
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
void apply_effects(const char *path, int *effects)
{
  // Setta o máximo de threads disponíveis
  omp_set_num_threads(omp_get_max_threads());

  // Lê a imagem ppm gerada
  PPMReader reader(path);
  char *filename = (char *)malloc(0);
  char *fullpath = (char *)malloc(0);

  // Aplica os efeitos
  while (*effects != INT_MIN)
  {
    switch (*effects)
    {
    case 1:
      reader.load();
      reverseColor(reader.getImage());
      fullpath = read_str("img/");
      filename = get_filename(path);
      strcat(filename, "_1.ppm");
      strcat(fullpath, filename);
      reader.write(fullpath);
      break;
    case 2:
      reader.load();
      green(reader.getImage());
      fullpath = read_str("img/");
      filename = get_filename(path);
      strcat(filename, "_2.ppm");
      strcat(fullpath, filename);
      reader.write(fullpath);
      break;
    case 3:
      reader.load();
      striped(reader.getImage());
      fullpath = read_str("img/");
      filename = get_filename(path);
      strcat(filename, "_3.ppm");
      strcat(fullpath, filename);
      reader.write(fullpath);
      break;
    default:
      cout << "O efeito '" << *effects << "' não foi encontrado." << endl;
    }

    // Para o otimizador do compilador não achar que essa variável não serve para nada
    if (*(effects++))
    {
    };
  }

  free(fullpath);
  free(filename);
}

void reverseColor(PPMImage *img)
{
#pragma omp parallel for
  for (int i = 0; i < img->width * img->height; i++)
  {
    img->pixel[i].r = img->maxColorVal - img->pixel[i].r;
    img->pixel[i].g = img->maxColorVal - img->pixel[i].g;
    img->pixel[i].b = img->maxColorVal - img->pixel[i].b;
  }
}

void green(PPMImage *img)
{
#pragma omp parallel for
  for (int i = 0; i < img->width * img->height; i++)
  {
    img->pixel[i].r = (int)img->pixel[i].r * 0.2126;
    img->pixel[i].g = (int)img->pixel[i].g * 0.7152;
    img->pixel[i].b = (int)img->pixel[i].b * 0.0722;
  }
}

void striped(PPMImage *img)
{
#pragma omp parallel for
  for (int i = 0; i < img->width * img->height; i++)
  {
    if (i % 3 == 0)
    {
      img->pixel[i].r = img->maxColorVal - img->pixel[i].r;
      img->pixel[i].g = img->maxColorVal - img->pixel[i].g;
      img->pixel[i].b = img->maxColorVal - img->pixel[i].b;
    }
  }
}

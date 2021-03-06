#ifndef FILETOOLS_H_INCLUDED
#define FILETOOLS_H_INCLUDED

#define FILE_PATH "/home/ricardo/Desktop/a.jpg"

#include "PPMImage.h"

/* Extrai o nome de um arquivo, se a sua extensão, de um caminho informado. */
char * get_filename(const char *);

/* Extrai os efeitos passados em uma string. */
int * extract_effects(const char *);

/* Remove os espaços à esquerda de uma string. */
char * ltrim(char *);

/* Remove os espaços à direita de uma string. */
char * rtrim(char *);

/* Remove os espaços de uma string. */
char * trim(char *);

/* Aplica os efeitos na imagem */
void apply_effects(const char *, int *);

void reverseColor(PPMImage *);

void green(PPMImage *);

void striped(PPMImage *);

#endif // FILETOOLS_H_INCLUDED

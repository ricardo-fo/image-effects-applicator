#include <iostream>
#include <cstring>
#include "utils.h"

using namespace std;

/**
 * Function: read_path
 * --------------------------------------
 * Lê uma string e aloca-a dinamicamente, retornando seu endereço na memória.
 *
 * param: str - String a ser alocada dinamicamente.
 *
 * returns: char *
 */
char * read_str(string str) {
  if (str.length() == 0) {
      getline(cin, str);
  }

  // Converte a string para char *
  char * c_str = new char [str.length() + 1];
  strcpy(c_str, str.c_str());

  // Aloca dinamicamente espaço para a string
  void * dyn_str = (void *) malloc (sizeof(char));
  dyn_str = realloc(c_str, (str.length() + 1) * sizeof(char));

  return (char *) dyn_str;
}

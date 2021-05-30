#include <iostream>
#include "utils.h"
#include "filetools.h"

using namespace std;

int main() {
    // Lê o caminho até o arquivo
    cout << "Caminho absoluto até a sua imagem: ";
    char * path = read_str(FILE_PATH);

    // Checa se o arquivo existe
    if (!has_file(path)) {
      cout << "\nO arquivo do caminho informado não pôde ser encontrado." << endl;
      return 1;
    }

    // Transforma o arquivo para .ppm
    char * filename = to_ppm(path);
    cout << "Seu arquivo: " << filename << endl;

    return 0;
}

#include <iostream>
#include "utils.h"
#include "filetools.h"
#include "Classes/PPMImage.h"

using namespace std;

int main() {
    PPMImage p;
    // Lê o caminho até o arquivo
    cout << "Caminho absoluto até a sua imagem: ";
    char * path = read_str(FILE_PATH);

    // Checa se o arquivo existe
    if (!has_file(path)) {
      cout << "\nO arquivo do caminho informado não pôde ser encontrado." << endl;
      return 1;
    }

    // Transforma o arquivo para .ppm
    char * new_path = to_ppm(path);

    // Aplica os efeitos e salva as imagens
    apply_effects(new_path);

    return 0;
}

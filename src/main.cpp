#include <iostream>
#include <climits>
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
    char * new_path = to_ppm(path);
    free(path);

    // Aplica os efeitos e salva as imagens
    cout << "\nDigite o número dos efeitos que deseja aplicar, seperando-os por vírgula.\nExemplo:\n>>> 1, 2\n>>> 3, 1, 2\n\nEfeitos disponíveis:\n<1> Inverter cores;\n<2> Sépia;\n<3> Granulado.\n>>> ";
    char * effects = read_str();
    int * effectsArr = extract_effects(effects);
    free(effects);

    apply_effects(new_path, effectsArr);

    cout << "\nEfeitos aplicados com sucesso." << endl;

    return 0;
}

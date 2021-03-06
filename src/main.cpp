#include <iostream>
#include <climits>
#include <cuda.h>
#include "utils.h"
#include "filetools.h"

using namespace std;

int main() {
    // Lê o caminho até o arquivo
    cout << "Caminho absoluto até a sua imagem: ";
    char * path = read_str();

    // Checa se o arquivo existe
    

    // Aplica os efeitos e salva as imagens
    cout << "\nDigite o número dos efeitos que deseja aplicar, seperando-os por vírgula.\nExemplo:\n>>> 1, 2\n>>> 3, 1, 2\n\nEfeitos disponíveis:\n<1> Inverter cores;\n<2> Verde;\n<3> Listrado.\n>>> ";
    char * effects = read_str();
    int * effectsArr = extract_effects(effects);
    free(effects);

    apply_effects(path, effectsArr);

    cout << "\nEfeitos aplicados com sucesso." << endl;

    return 0;
}

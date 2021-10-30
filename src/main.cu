#include <iostream>
#include <climits>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;


char * read_str() {
  string str;
  getline(cin, str);

  // Converte a string para char *
  char * c_str = new char [str.length() + 1];
  strcpy(c_str, str.c_str());

  // Aloca dinamicamente espaço para a string
  char * dyn_str;
  cudaMalloc(&dyn_str, sizeof(char));
  dyn_str = (char *) realloc(c_str, (str.length() + 1) * sizeof(char));

  return dyn_str;
}

char * read_str(string str) {
  if (str.length() == 0) {
      getline(cin, str);
  }

  // Converte a string para char *
  char * c_str = new char [str.length() + 1];
  strcpy(c_str, str.c_str());

  // Aloca dinamicamente espaço para a string
  char * dyn_str;
  cudaMalloc(&dyn_str, sizeof(char));
  dyn_str = (char *) realloc(c_str, (str.length() + 1) * sizeof(char));

  return dyn_str;
}

typedef struct {
  unsigned char r, g, b;
} PPMPixel;

class PPMImage {
  public:
    PPMImage();

    int width; // Width of the image
    int height; // Height of the image
    int maxColorVal; // Maximum color value
    PPMPixel * pixel;
};

class PPMReader {
  private:
    char * path;
    PPMImage * image;

  public:
    PPMReader(const char *);

    // Carrega a imagem ppm
    void load();

    // Salva uma imagem ppm
    void write(const char *);

    // Retorna um objeto contendo os dados da imagem PPM
    PPMImage * getImage();

    void setImage(PPMImage *);

    // Muda o caminho até o arquivo ppm
    void setPath(const char *);
};

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
  cudaMalloc(&img, sizeof(PPMImage));
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

  cudaMalloc(&img->pixel, sizeof(PPMPixel) * img->width * img->height);
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

void PPMReader::setImage(PPMImage * img) {
  cudaMalloc(&image, sizeof(img));
  image = img;
}

void PPMReader::setPath(const char * _path) {
  path = new char[strlen(_path) + 1];
  strcpy(path, _path);
}

char *get_filename(const char *path)
{
  string cpp_path = string(path);
  size_t last_bar = cpp_path.find_last_of("/");
  size_t last_dot = cpp_path.find_last_of(".");
  string filename = cpp_path.substr(last_bar + sizeof(char), last_dot - last_bar - 1);

  return read_str(filename);
}

char *ltrim(char *str)
{
  while (isspace(*str))
    str++;
  return str;
}


char *rtrim(char *str)
{
  char *back = str + strlen(str);
  while (isspace(*--back))
    ;
  *(back + 1) = '\0';
  return str;
}

char *trim(char *str)
{
  return rtrim(ltrim(str));
}


int *extract_effects(const char *effects)
{
  const char *delim = ",";
  char *copy = new char[strlen(effects) + 1];    // Cópia da string effects para ser manipulada
  char *token = new char[strlen(effects) + 1];   // Token, i.e. efeitos
  char *sanitized = new char[strlen(token) + 1]; // Token sanitizado
  int *effects_arr;
  cudaMalloc(&effects_arr, 0);           // Reserva espaço na memória para o vetor de efeitos

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

__global__ void reverseColor(PPMImage *img)
{
//#pragma omp parallel for
  for (int i = 0; i < img->width * img->height; i++)
  {
    img->pixel[i].r = img->maxColorVal - img->pixel[i].r;
    img->pixel[i].g = img->maxColorVal - img->pixel[i].g;
    img->pixel[i].b = img->maxColorVal - img->pixel[i].b;
  }
}

__global__ void green(PPMImage *img)
{
//#pragma omp parallel for
  for (int i = 0; i < img->width * img->height; i++)
  {
    img->pixel[i].r = (int)img->pixel[i].r * 0.2126;
    img->pixel[i].g = (int)img->pixel[i].g * 0.7152;
    img->pixel[i].b = (int)img->pixel[i].b * 0.0722;
  }
}

__global__ void striped(PPMImage *img)
{
//#pragma omp parallel for
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

void apply_effects()
{
  char * path = "/home/fabio/image-effects-applicator/src/img/vistaSuperior.ppm";
  int effects[1] = {1};
  printf("chegou aqui");

  PPMReader reader(path);
  char *filename;
  char *fullpath;
  cudaMalloc(&filename, 0);
  cudaMalloc(&fullpath, 0);

  // // Aplica os efeitos
  while (*effects != INT_MIN)
  {
    switch (*effects)
    {
    case 1:
      reader.load();
      reverseColor<<<1,1>>>(reader.getImage());
      fullpath = read_str("img/");
      filename = get_filename(path);
      strcat(filename, "_1.ppm");
      strcat(fullpath, filename);
      reader.write(fullpath);
      break;
    case 2:
      reader.load();
      green<<<1,1>>>(reader.getImage());
      fullpath = read_str("img/");
      filename = get_filename(path);
      strcat(filename, "_2.ppm");
      strcat(fullpath, filename);
      reader.write(fullpath);
      break;
    case 3:
      reader.load();
      striped<<<1,1>>>(reader.getImage());
      fullpath = read_str("img/");
      filename = get_filename(path);
      strcat(filename, "_3.ppm");
      strcat(fullpath, filename);
      reader.write(fullpath);
      break;
    default:
      cout << "O efeito '" << *effects << "' não foi encontrado." << endl;
    }
  }

  // cudaFree(fullpath);
  // cudaFree(filename);
}


int main() {
    // Lê o caminho até o arquivo
    cout << "Caminho absoluto até a sua imagem: ";
    char * path = read_str();

    // Checa se o arquivo existe
    

    // Aplica os efeitos e salva as imagens
    cout << "\nDigite o número dos efeitos que deseja aplicar, seperando-os por vírgula.\nExemplo:\n>>> 1, 2\n>>> 3, 1, 2\n\nEfeitos disponíveis:\n<1> Inverter cores;\n<2> Verde;\n<3> Listrado.\n>>> ";
    char * effects = read_str();
    int * effectsArr = extract_effects(effects);

    printf("%s\n", path);
    printf("%d\n", effectsArr[0]);
    apply_effects();

    cout << "\nEfeitos aplicados com sucesso." << endl;

    return 0;
}

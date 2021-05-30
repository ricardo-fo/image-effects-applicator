#ifndef FILETOOLS_H_INCLUDED
#define FILETOOLS_H_INCLUDED

#define FILE_PATH "/home/ricardo/Pictures/Wallpapers/pink_floyd.jpg"

/* Verifica se um arquivo existe. */
bool has_file(const char *);

/* Extraí o nome de um arquivo, se a sua extensão, de um caminho informado. */
char * get_filename(const char *);

/* Converte um arquivo de imagem para .ppm */
char * to_ppm(const char *);

#endif // FILETOOLS_H_INCLUDED

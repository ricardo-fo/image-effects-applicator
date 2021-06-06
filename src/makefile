##### Variáveis #####
# Compilador
CC=g++

# Flags
CFLAGS=-W -Wall -O

# OpenMP
OMPFLAGS=-L/usr/X11R6/lib -lm -lpthread -lX11 -fopenmp

# Executável
EXEC=exec

# Nome do executável
PROJ_NAME=exec

# Arquivos source, i.e. extensão .c
SRC=$(wildcard *.cpp)

# Arquivos objeto, i.e. extensão .o
OBJ=$(SRC:.cpp=.o)

##### Regras e dependências #####
$(PROJ_NAME): $(OBJ)
	$(CC) -o $(PROJ_NAME) $(OBJ) $(OMPFLAGS) $(CFLAGS) -x c++
	@echo "\nCompilação finalizada!"
	@echo "\nComandos aceitos:"
	@echo "'make run'      - Executa a aplicação."
	@echo "'make free'     - Remove todos os arquivos .ppm que estão no diretório 'img/'."
	@echo "'make clean'    - Remove todos os arquivos .o que foram gerados."
	@echo "'make mrproper' - Remove o arquivo executável gerado."

main.o: main.cpp filetools.h utils.h CImg.h
	$(CC) -c main.cpp $(CFLAGS)

filetools.o: filetools.h utils.h CImg.h PPMImage.h PPMReader.h
	$(CC) -c filetools.cpp $(CFLAGS) $(OMPFLAGS)

utils.o: utils.h
	$(CC) -c utils.cpp $(CFLAGS)

PPMImage.o: PPMImage.h
	$(CC) -c PPMImage.cpp $(CFLAGS)

PPMReader.o: PPMReader.h PPMImage.h
	$(CC) -c PPMReader.cpp $(CFLAGS)

# Executar
run:
	./$(EXEC)

# Limpa as imagens
free:
	rm -rf img/*.ppm

# Limpar de arquivos objeto
clean:
	rm -rf $(OBJ)

# Apagar tudo o que deve ser modificado
mrproper:
	rm -rf $(EXEC)

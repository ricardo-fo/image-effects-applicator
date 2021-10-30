##### Variáveis #####
# Compilador
CC=nvcc

# Flags
#CFLAGS=--compiler-options -W --compiler-options -Wall --compiler-options -O  --compiler-options -lpng --compiler-options -Dcimg_display=0 --compiler-options -Dcimg_use_vt100
#XFLAGS=-Wno-deprecated-gpu-targets --compiler-options -Wextra --compiler-options -Wall --compiler-options -O3 --compiler-options -Wno-unused-result --compiler-options -Wno-unused-parameter

# OpenMP
#OMPFLAGS= --compiler-options -L/usr/X11R6/lib --compiler-options -lm  -lpthread  -lX11 --compiler-options -fopenmp

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
	$(CC) -o $(PROJ_NAME) $(OBJ) 
	@echo "\nCompilação finalizada!"
	@echo "\nComandos aceitos:"
	@echo "'make run'      - Executa a aplicação."
	@echo "'make free'     - Remove todos os arquivos .ppm que estão no diretório 'img/'."
	@echo "'make clean'    - Remove todos os arquivos .o que foram gerados."
	@echo "'make mrproper' - Remove o arquivo executável gerado."

main.o: main.cpp filetools.h utils.h
	$(CC) -c main.cpp 

filetools.o: filetools.h utils.h PPMImage.h PPMReader.h
	$(CC) -c filetools.cpp 

utils.o: utils.h
	$(CC) -c utils.cpp 

PPMImage.o: PPMImage.h
	$(CC) -c PPMImage.cpp

PPMReader.o: PPMReader.h PPMImage.h
	$(CC) -c PPMReader.cpp 

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

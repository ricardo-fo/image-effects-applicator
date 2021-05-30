##### Variáveis #####
# Compilador
CC=g++

# Flags
CFLAGS=-W -Wall -L/usr/X11R6/lib -lm -lpthread -lX11

# Executável
EXEC=aplicacao.exe

# File
FILE=main.cpp

# Nome do executavel
PROJ_NAME=aplicacao.exe

# Arquivos source, i.e. extensão .c
C_SRC=$(wildcard *.cpp)

# Arquivos objeto, i.e. extensão .o
OBJ=$(C_SRC:.c=.o)

##### Regras e dependências #####
$(PROJ_NAME): $(OBJ)
	$(CC) -o $(PROJ_NAME) $(OBJ) $(CFLAGS) -lm
	@echo "\nCompilação finalizada!"
	@echo "\nComandos aceitos:"
	@echo "'make run'      - Executa a aplicação."
	@echo "'make clean'    - Remove todos os arquivos .ppm que estão no mesmo diretório da aplicação."
	@echo "'make mrproper' - Remove o arquivo executável gerado."

main.o: main.c filetools.h utils.h CImg.h
	$(CC) -cpp $(FILE) $(CFLAGS)

filetools.o: filetools.h utils.h
	$(CC) -cpp filetools.cpp $(CFLAGS)

utils.o: utils.h
	$(CC) -cpp utils.cpp $(CFLAGS)

# Executar
run:
	./$(EXEC)

# Limpar de arquivos
clean:
	rm -rf *.ppm

# Apagar tudo o que deve ser modificado
mrproper:
	rm -rf $(EXEC)

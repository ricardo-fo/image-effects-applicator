##### Variáveis #####
# Compilador
CC=g++

# Flags
CFLAGS=-W -Wall -O2 -L/usr/X11R6/lib -lm -lpthread -lX11

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
	@echo "\nCompilação finalizada!\nDigite 'make run' e tecle <enter> para executar a aplicacao"

main.o: main.c CImg.h
	$(CC) -c $(FILE) $(CFLAGS)

# Executar
run:
	./$(EXEC)

# Limpar de arquivos
clean:
	rm -rf *.o

# Apagar tudo o que deve ser modificado
mrproper:
	rm -rf $(EXEC)

# Makefile para compilación de multiplicación de matrices con CUDA y OpenMP
# Autor: Tarea Universitaria - Programación Paralela
# Hardware objetivo: NVIDIA RTX 2070 Super (Compute Capability 7.5)

# Compilador CUDA
NVCC = nvcc

# Flags de compilación
CFLAGS = -O3 -arch=sm_75 -Xcompiler -fopenmp

# Nombre del archivo fuente
SRC = matmul_compa.cu

# Nombre del ejecutable
TARGET = prog

# Regla por defecto
all: $(TARGET)

# Regla de compilación
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

# Regla para limpiar archivos generados
clean:
	rm -f $(TARGET)

# Regla para recompilar desde cero
rebuild: clean all

# Regla de ayuda
help:
	@echo "Uso del Makefile:"
	@echo "  make         - Compila el programa"
	@echo "  make clean   - Elimina el ejecutable"
	@echo "  make rebuild - Recompila desde cero"
	@echo "  make help    - Muestra esta ayuda"

.PHONY: all clean rebuild help

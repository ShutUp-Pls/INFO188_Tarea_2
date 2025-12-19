# Makefile para compilación de multiplicación de matrices con CUDA y OpenMP
# Autor: Tarea Universitaria - Programación Paralela
# Hardware esperado:
# - Por defecto [Pascal] (Compute Capability 6.0)
# - NVIDIA RTX 2070 Super [Turing] (Compute Capability 7.5)
# - NVIDIA RTX 3060 Laptop [Ampere] (Compute Capability 8.6)

# Compilador CUDA
NVCC = nvcc

# Nombre del archivo fuente y ejecutable
SRC = matmul_compa.cu
TARGET = prog

# Intentamos obtener el nombre de la GPU usando nvidia-smi
DETECTED_GPU := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)

# Definimos como Fallback sm_60 (Pascal [Compute Capability 6.0])
ARCH_FLAG := -arch=sm_60
MSG_GPU := GPU no identificada o específica. Usando Fallback (sm_60).

# Si detecta "2070" (Turing)
ifneq (,$(findstring 2070,$(DETECTED_GPU)))
    ARCH_FLAG := -arch=sm_75
    MSG_GPU := Configurando para Turing (sm_75)
endif

# Si detecta "3060" (Ampere)
ifneq (,$(findstring 3060,$(DETECTED_GPU)))
    ARCH_FLAG := -arch=sm_86
    MSG_GPU := Configurando para Ampere (sm_86)
endif

# Flags de compilación
CFLAGS = -O3 $(ARCH_FLAG) -Xcompiler -fopenmp

# Regla por defecto
all: info $(TARGET)

# Regla de compilación
$(TARGET): $(SRC)
	@echo "Compilando para: $(ARCH_FLAG)"
	$(NVCC) $(CFLAGS) $(SRC) -o $(TARGET)

# Regla para limpiar
clean:
	rm -f $(TARGET)

# Regla para recompilar
rebuild: clean all

# Regla informativa (Que estamos detectando)
info:
	@echo "----------------------------------------------------------"
	@echo "Información de Hardware:"
	@echo "  GPU Detectada por OS : $(DETECTED_GPU)"
	@echo "  Configuración elegida: $(MSG_GPU)"
	@echo "  Flags de compilación : $(CFLAGS)"
	@echo "----------------------------------------------------------"

# Regla que verifica dependencias, librerías de Python y archivos fuente.
check:
	@echo "----------------------------------------------------------"
	@echo "Verificando dependencias y entorno..."
	@echo "----------------------------------------------------------"
	@# 1. Verificar NVCC
	@which $(NVCC) > /dev/null && echo " [OK] Compilador CUDA ($(NVCC))" || (echo " [ERROR] No se encontró $(NVCC). Instala CUDA Toolkit."; exit 1)
	@# 2. Verificar conexión con GPU
	@nvidia-smi > /dev/null 2>&1 && echo " [OK] Driver NVIDIA y GPU detectados" || (echo " [ERROR] 'nvidia-smi' falló. Verifica tus drivers."; exit 1)
	@# 3. Verificar Python 3
	@which python3 > /dev/null && echo " [OK] Python 3" || (echo " [ERROR] No se encontró python3."; exit 1)
	@# 4. Verificar Librerías Python (Numpy)
	@python3 -c "import numpy" > /dev/null 2>&1 && echo " [OK] Python Library: numpy" || (echo " [ERROR] Falta librería 'numpy'. Ejecuta: pip install numpy"; exit 1)
	@# 5. Verificar Librerías Python (Matplotlib)
	@python3 -c "import matplotlib" > /dev/null 2>&1 && echo " [OK] Python Library: matplotlib" || (echo " [ERROR] Falta librería 'matplotlib'. Ejecuta: pip install matplotlib"; exit 1)
	@# 6. Verificar Archivos Fuente
	@[ -f $(SRC) ] && echo " [OK] Archivo fuente C++ ($(SRC))" || (echo " [ERROR] Falta el archivo $(SRC)."; exit 1)
	@[ -f $(PY_SCRIPT) ] && echo " [OK] Script Python ($(PY_SCRIPT))" || (echo " [ERROR] Falta el archivo $(PY_SCRIPT)."; exit 1)
	@echo "----------------------------------------------------------"
	@echo "Todo listo. Puedes ejecutar 'make' para compilar."

# Regla de ayuda
help:
	@echo "Uso del Makefile:"
	@echo "  make         - Detecta GPU y compila"
	@echo "  make check   - Verifica compiladores, librerías python y archivos"
	@echo "  make info    - Solo muestra qué GPU se detectó sin compilar"
	@echo "  make clean   - Elimina el ejecutable y gráficos generados"
	@echo "  make rebuild - Recompila desde cero"

.PHONY: all clean rebuild help info check
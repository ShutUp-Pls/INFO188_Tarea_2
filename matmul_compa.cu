/*
 * Programa de Comparación de Multiplicación de Matrices
 * Implementa 3 algoritmos: CPU OpenMP, GPU Global Memory, GPU Shared Memory
 * Uso: ./matmul_compa <n> <nt> <ALG>
 *   n: Tamaño de la matriz (NxN)
 *   nt: Número de threads CPU (para OpenMP)
 *   ALG: 1=CPU Multicore, 2=GPU Básica, 3=GPU Shared Memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <time.h>

#define TILE_WIDTH 16

// ============================================================================
// KERNELS DE GPU
// ============================================================================

// ALG 2: GPU Básica (Global Memory)
__global__ void kernel_matmul(int n, float *a, float *b, float *c) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;
    
    if(tx < n && ty < n) {
        for(int k = 0; k < n; ++k) {
            sum += a[ty * n + k] * b[k * n + tx];
        }
        c[ty * n + tx] = sum;
    }
}

// ALG 3: GPU Shared Memory (Tiled Matrix Multiplication)
__global__ void kernel_matmul_tiled(int n, float *a, float *b, float *c) {
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    // Itera sobre los tiles
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for(int t = 0; t < numTiles; ++t) {
        // Carga tile de A en memoria compartida
        int a_col = t * TILE_WIDTH + tx;
        if(row < n && a_col < n) {
            tile_a[ty][tx] = a[row * n + a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }
        
        // Carga tile de B en memoria compartida
        int b_row = t * TILE_WIDTH + ty;
        if(b_row < n && col < n) {
            tile_b[ty][tx] = b[b_row * n + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Calcula producto parcial
        for(int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }
        
        __syncthreads();
    }
    
    // Escribe resultado
    if(row < n && col < n) {
        c[row * n + col] = sum;
    }
}

// ============================================================================
// FUNCIONES AUXILIARES
// ============================================================================

// Inicializa matriz con valores aleatorios
void initialize_matrix(float *mat, int n) {
    for(int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

// Verifica errores de CUDA
void check_cuda_error(cudaError_t err, const char *msg) {
    if(err != cudaSuccess) {
        fprintf(stderr, "Error CUDA en %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Muestra información del dispositivo GPU
void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("========================================\n");
    printf("Dispositivo GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memoria Global: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("========================================\n\n");
}

// ============================================================================
// ALGORITMOS DE MULTIPLICACIÓN
// ============================================================================

// ALG 1: CPU Multicore (OpenMP)
void matmul_cpu_openmp(float *a, float *b, float *c, int n, int nt) {
    omp_set_num_threads(nt);
    
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    
    printf("Algoritmo: CPU Multicore (OpenMP)\n");
    printf("Threads CPU: %d\n", nt);
    printf("Tiempo de ejecución: %.6f segundos\n", elapsed);
}

// ALG 2: GPU Básica (Global Memory)
void matmul_gpu_global(float *h_a, float *h_b, float *h_c, int n) {
    size_t size = n * n * sizeof(float);
    
    // Reservar memoria en device
    float *d_a, *d_b, *d_c;
    check_cuda_error(cudaMalloc(&d_a, size), "cudaMalloc d_a");
    check_cuda_error(cudaMalloc(&d_b, size), "cudaMalloc d_b");
    check_cuda_error(cudaMalloc(&d_c, size), "cudaMalloc d_c");
    
    // Copiar datos de host a device
    check_cuda_error(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
    check_cuda_error(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");
    
    // Configurar grid y bloques
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    
    // Crear eventos para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Ejecutar kernel
    cudaEventRecord(start);
    kernel_matmul<<<gridDim, blockDim>>>(n, d_a, d_b, d_c);
    cudaEventRecord(stop);
    
    check_cuda_error(cudaGetLastError(), "kernel_matmul launch");
    cudaEventSynchronize(stop);
    
    // Calcular tiempo
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copiar resultado de device a host
    check_cuda_error(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");
    
    printf("Algoritmo: GPU Básica (Global Memory)\n");
    printf("Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("Tiempo de ejecución: %.6f segundos\n", milliseconds / 1000.0f);
    
    // Liberar memoria
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ALG 3: GPU Shared Memory (Tiled)
void matmul_gpu_tiled(float *h_a, float *h_b, float *h_c, int n) {
    size_t size = n * n * sizeof(float);
    
    // Reservar memoria en device
    float *d_a, *d_b, *d_c;
    check_cuda_error(cudaMalloc(&d_a, size), "cudaMalloc d_a");
    check_cuda_error(cudaMalloc(&d_b, size), "cudaMalloc d_b");
    check_cuda_error(cudaMalloc(&d_c, size), "cudaMalloc d_c");
    
    // Copiar datos de host a device
    check_cuda_error(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
    check_cuda_error(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");
    
    // Configurar grid y bloques
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Crear eventos para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Ejecutar kernel
    cudaEventRecord(start);
    kernel_matmul_tiled<<<gridDim, blockDim>>>(n, d_a, d_b, d_c);
    cudaEventRecord(stop);
    
    check_cuda_error(cudaGetLastError(), "kernel_matmul_tiled launch");
    cudaEventSynchronize(stop);
    
    // Calcular tiempo
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copiar resultado de device a host
    check_cuda_error(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");
    
    printf("Algoritmo: GPU Shared Memory (Tiled)\n");
    printf("Tile Width: %d\n", TILE_WIDTH);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("Tiempo de ejecución: %.6f segundos\n", milliseconds / 1000.0f);
    
    // Liberar memoria
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// FUNCIÓN PRINCIPAL
// ============================================================================

int main(int argc, char *argv[]) {
    // Validar argumentos
    if(argc != 4) {
        fprintf(stderr, "Uso: %s <n> <nt> <ALG>\n", argv[0]);
        fprintf(stderr, "  n:   Tamaño de la matriz (NxN)\n");
        fprintf(stderr, "  nt:  Número de threads CPU (para OpenMP)\n");
        fprintf(stderr, "  ALG: 1=CPU Multicore, 2=GPU Básica, 3=GPU Shared Memory\n");
        return EXIT_FAILURE;
    }
    
    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);
    
    if(n <= 0 || nt <= 0 || alg < 1 || alg > 3) {
        fprintf(stderr, "Error: Parámetros inválidos\n");
        return EXIT_FAILURE;
    }
    
    printf("\n========================================\n");
    printf("MULTIPLICACIÓN DE MATRICES\n");
    printf("========================================\n");
    printf("Tamaño de matriz: %d x %d\n", n, n);
    printf("Elementos totales: %d\n", n * n);
    printf("========================================\n\n");
    
    // Reservar memoria para matrices en host
    size_t size = n * n * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    if(!h_a || !h_b || !h_c) {
        fprintf(stderr, "Error: No se pudo reservar memoria en host\n");
        return EXIT_FAILURE;
    }
    
    // Inicializar matrices con valores aleatorios
    srand(time(NULL));
    initialize_matrix(h_a, n);
    initialize_matrix(h_b, n);
    
    // Ejecutar algoritmo seleccionado
    switch(alg) {
        case 1:
            printf("Hardware: CPU con OpenMP\n");
            printf("========================================\n");
            matmul_cpu_openmp(h_a, h_b, h_c, n, nt);
            break;
            
        case 2:
            print_device_info();
            matmul_gpu_global(h_a, h_b, h_c, n);
            break;
            
        case 3:
            print_device_info();
            matmul_gpu_tiled(h_a, h_b, h_c, n);
            break;
            
        default:
            fprintf(stderr, "Error: Algoritmo no válido\n");
            free(h_a);
            free(h_b);
            free(h_c);
            return EXIT_FAILURE;
    }
    
    printf("========================================\n");
    printf("Verificación de resultados:\n");
    printf("C[0][0] = %.2f\n", h_c[0]);
    printf("C[%d][%d] = %.2f\n", n/2, n/2, h_c[(n/2) * n + (n/2)]);
    printf("C[%d][%d] = %.2f\n", n-1, n-1, h_c[(n-1) * n + (n-1)]);
    printf("========================================\n\n");
    
    // Liberar memoria
    free(h_a);
    free(h_b);
    free(h_c);
    
    return EXIT_SUCCESS;
}

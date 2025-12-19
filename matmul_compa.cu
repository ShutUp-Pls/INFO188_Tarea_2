/*
 * Programa de Comparación de Multiplicación de Matrices
 * Implementa 3 algoritmos: CPU OpenMP, GPU Global Memory, GPU Shared Memory
 * Uso: ./prog <n> <nt> <ALG>
 *   n: Tamaño de la matriz (NxN)
 *   nt: Número de threads CPU (para OpenMP)
 *   ALG: 1=CPU Multicore, 2=GPU Básica, 3=GPU Shared Memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <omp.h>
#include <time.h>

#define TILE_WIDTH 16
#define WARP_SIZE 32

using namespace nvcuda;

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

// ALG 4: GPU Tensor Cores (WMMA)
// Helper: Conversión Float -> Half
__global__ void float_to_half_kernel(float* f, half* h, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        h[idx] = __float2half(f[idx]);
    }
}

// Kernel WMMA: Cada Warp calcula un tile de 16x16
__global__ void kernel_matmul_wmma(half *a, half *b, float *c, int n) {
    // Declaración de fragmentos
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Inicializar acumulador a 0
    wmma::fill_fragment(c_frag, 0.0f);

    // Coordenadas globales del Warp
    // blockDim.y = 4 (4 warps por bloque), threadIdx.y es el índice del warp dentro del bloque (0..3)
    // blockIdx.y es el índice del bloque vertical
    int globalWarpM = blockIdx.y * blockDim.y + threadIdx.y;
    int globalWarpN = blockIdx.x; 

    // Requerimos que N sea múltiplo de 16 para este kernel simple
    // Iteramos sobre K en pasos de 16
    for (int i = 0; i < n; i += 16) {
        // Coordenadas base del tile actual en A y B
        int aRow = globalWarpM * 16;
        int aCol = i;
        
        int bRow = i;
        int bCol = globalWarpN * 16;
        
        // Bounds checking simple
        if (aRow < n && bCol < n) {
            // Cargar A
            const half* a_ptr = a + aRow * n + aCol;
            wmma::load_matrix_sync(a_frag, a_ptr, n);
            
            // Cargar B
            const half* b_ptr = b + bRow * n + bCol;
            wmma::load_matrix_sync(b_frag, b_ptr, n);
            
            // Multiplicar y acumular
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Almacenar resultado
    int cRow = globalWarpM * 16;
    int cCol = globalWarpN * 16;
    
    if (cRow < n && cCol < n) {
        float* c_ptr = c + cRow * n + cCol;
        wmma::store_matrix_sync(c_ptr, c_frag, n, wmma::mem_row_major);
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


// A partir de aquí se definen prints para quitar
// verbosidad en la funcionalidad de los algoritmos
void print_separator() {
    printf("========================================\n");
}

// Muestra información del dispositivo GPU
void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    print_separator();
    printf("Dispositivo GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memoria Global: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    print_separator();
}

void print_header(const char* title) {
    print_separator();
    printf("%s\n", title);
    print_separator();
}

void print_sub_header(const char* title) {
    printf("\n");
    print_separator();
    printf("%s\n", title);
    print_separator();
}

void print_matrix_config(int n) {
    printf("Tamaño de matriz: %d x %d\n", n, n);
    printf("Elementos totales: %d\n", n * n);
    print_separator();
    printf("\n");
}

void print_cpu_stats(int nt, double elapsed) {
    printf("Algoritmo: CPU Multicore (OpenMP)\n");
    printf("Threads CPU: %d\n", nt);
    printf("Tiempo de ejecución: %.6f segundos\n", elapsed);
}

void print_gpu_stats(const char* algo_name, dim3 grid, dim3 block, float milliseconds) {
    printf("Algoritmo: %s\n", algo_name);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    printf("Tiempo de ejecución: %.6f segundos\n", milliseconds / 1000.0f);
}

void print_tiled_stats(int tile_w, dim3 grid, dim3 block, float milliseconds) {
    printf("Algoritmo: GPU Shared Memory (Tiled)\n");
    printf("Tile Width: %d\n", tile_w);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    printf("Tiempo de ejecución: %.6f segundos\n", milliseconds / 1000.0f);
}

void print_verification_result(float *h_c, int n) {
    print_separator();
    printf("Verificación de resultados:\n");
    printf("C[0][0] = %.2f\n", h_c[0]);
    printf("C[%d][%d] = %.2f\n", n/2, n/2, h_c[(n/2) * n + (n/2)]);
    printf("C[%d][%d] = %.2f\n", n-1, n-1, h_c[(n-1) * n + (n-1)]);
    print_separator();
    printf("\n");
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
    
    print_cpu_stats(nt, elapsed);
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
    
    print_gpu_stats("GPU Básica (Global Memory)", gridDim, blockDim, milliseconds);
    
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
    
    print_tiled_stats(TILE_WIDTH, gridDim, blockDim, milliseconds);
    
    // Liberar memoria
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ALG 4: GPU Tensor Cores (WMMA)
void matmul_gpu_tc(float *h_a, float *h_b, float *h_c, int n) {
    // Validar N (WMMA simple prefiere múltiplos de 16)
    if (n % 16 != 0) {
        printf("Advertencia: Para Tensor Cores simple, N debe ser múltiplo de 16. Padding virtual no implementado.\n");
    }

    size_t size_float = n * n * sizeof(float);
    size_t size_half = n * n * sizeof(half);
    
    float *d_c;
    half *d_a_half, *d_b_half;
    float *d_a_temp, *d_b_temp; // Temporales para subida y conversión

    check_cuda_error(cudaMalloc(&d_c, size_float), "cudaMalloc d_c");
    check_cuda_error(cudaMalloc(&d_a_half, size_half), "cudaMalloc d_a_half");
    check_cuda_error(cudaMalloc(&d_b_half, size_half), "cudaMalloc d_b_half");
    
    // Para ser justos, debemos convertir los datos a half.
    // Usamos punteros temporales float para subir los datos y luego convertirlos
    check_cuda_error(cudaMalloc(&d_a_temp, size_float), "Malloc temp A");
    check_cuda_error(cudaMalloc(&d_b_temp, size_float), "Malloc temp B");
    
    check_cuda_error(cudaMemcpy(d_a_temp, h_a, size_float, cudaMemcpyHostToDevice), "H2D A");
    check_cuda_error(cudaMemcpy(d_b_temp, h_b, size_float, cudaMemcpyHostToDevice), "H2D B");

    // Convertir Float a Half (Preparación de datos, no medido en el kernel de cómputo para justicia comparativa)
    int threads = 256;
    int blocks = (n * n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads>>>(d_a_temp, d_a_half, n);
    float_to_half_kernel<<<blocks, threads>>>(d_b_temp, d_b_half, n);
    check_cuda_error(cudaDeviceSynchronize(), "Float2Half Conv");
    
    // Liberar temporales
    cudaFree(d_a_temp);
    cudaFree(d_b_temp);

    // Configuración de Grid para WMMA
    // 1 Warp = 1 Tile (16x16)
    // Block size = 128 threads (4 warps)
    dim3 blockDim(32, 4); 
    // Grid: cubrimos N/16 tiles
    dim3 gridDim((n / 16 + (blockDim.x/32) - 1) / (blockDim.x/32), (n / 16 + blockDim.y - 1) / blockDim.y);
    // Ajuste simple: grid.x = Tiles en col, grid.y = Tiles en fila / warps_per_block
    gridDim.x = n / 16; 
    gridDim.y = (n / 16 + 3) / 4; // 4 warps en Y por bloque

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Medir SOLO el cómputo matricial
    cudaEventRecord(start);
    kernel_matmul_wmma<<<gridDim, blockDim>>>(d_a_half, d_b_half, d_c, n);
    cudaEventRecord(stop);
    
    check_cuda_error(cudaGetLastError(), "kernel_matmul_wmma launch");
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    check_cuda_error(cudaMemcpy(h_c, d_c, size_float, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");
    
    print_gpu_stats("GPU Tensor Cores (WMMA)", gridDim, blockDim, milliseconds);
    
    cudaFree(d_a_half); cudaFree(d_b_half); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
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
        fprintf(stderr, "  ALG: 1=CPU, 2=GPU Global, 3=GPU Shared, 4=GPU Tensor Cores\n");
        return EXIT_FAILURE;
    }
    
    int n = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int alg = atoi(argv[3]);
    
    if(n <= 0 || nt <= 0 || alg < 1 || alg > 4) {
        fprintf(stderr, "Error: Parámetros inválidos\n");
        return EXIT_FAILURE;
    }
    
    print_header("MULTIPLICACIÓN DE MATRICES");
    print_matrix_config(n);
    
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
            print_sub_header("Hardware: CPU con OpenMP");
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

        case 4:
            print_device_info();
            matmul_gpu_tc(h_a, h_b, h_c, n);
            break;
            
        default:
            fprintf(stderr, "Error: Algoritmo no válido\n");
            free(h_a);
            free(h_b);
            free(h_c);
            return EXIT_FAILURE;
    }
    
    print_verification_result(h_c, n);
    
    // Liberar memoria
    free(h_a);
    free(h_b);
    free(h_c);
    
    return EXIT_SUCCESS;
}

#!/usr/bin/env python3
"""
Script de Automatización de Benchmarks para Multiplicación de Matrices
Ejecuta pruebas de rendimiento y genera gráficos comparativos
Algoritmos: 1=CPU OpenMP, 2=GPU Global Memory, 3=GPU Shared Memory
"""

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Configuración de parámetros
N_SIZES = [512, 1024, 2048, 4096, 8192]  # Tamaños de matrices a probar
NUM_THREADS = 16  # Número de threads para OpenMP (ajustar según CPU)
EXECUTABLE = "./prog"
ALGORITHMS = {
    1: "CPU OpenMP",
    2: "GPU Global Memory",
    3: "GPU Shared Memory"
}

def check_executable():
    """Verifica que el ejecutable existe"""
    if not os.path.exists(EXECUTABLE):
        print(f"Error: No se encuentra el ejecutable '{EXECUTABLE}'")
        print("Por favor, ejecuta 'make' primero para compilar el programa.")
        sys.exit(1)

def run_benchmark(n, nt, alg):
    """
    Ejecuta el benchmark y extrae el tiempo de ejecución
    
    Args:
        n: Tamaño de la matriz
        nt: Número de threads CPU
        alg: Algoritmo (1, 2, o 3)
    
    Returns:
        float: Tiempo de ejecución en segundos (o None si hay error)
    """
    try:
        # Ejecutar el programa
        cmd = [EXECUTABLE, str(n), str(nt), str(alg)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Error ejecutando: {' '.join(cmd)}")
            print(f"stderr: {result.stderr}")
            return None
        
        # Parsear el tiempo de ejecución usando regex
        # Busca líneas como: "Tiempo de ejecución: 0.123456 segundos"
        pattern = r"Tiempo de ejecución:\s+([\d.]+)\s+segundos"
        match = re.search(pattern, result.stdout)
        
        if match:
            time_seconds = float(match.group(1))
            print(f"  N={n}, ALG={alg} ({ALGORITHMS[alg]}): {time_seconds:.6f} segundos")
            return time_seconds
        else:
            print(f"No se pudo extraer el tiempo de la salida para N={n}, ALG={alg}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout ejecutando N={n}, ALG={alg}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def collect_data():
    """
    Recolecta todos los datos de benchmark
    
    Returns:
        dict: Diccionario con los resultados para cada algoritmo
    """
    results = {alg: [] for alg in ALGORITHMS.keys()}
    
    print("="*60)
    print("INICIANDO BENCHMARKS DE MULTIPLICACIÓN DE MATRICES")
    print("="*60)
    print(f"Tamaños de matriz: {N_SIZES}")
    print(f"Threads CPU: {NUM_THREADS}")
    print("="*60)
    
    for n in N_SIZES:
        print(f"\nProbando N = {n}:")
        for alg in ALGORITHMS.keys():
            time = run_benchmark(n, NUM_THREADS, alg)
            results[alg].append(time)
    
    print("\n" + "="*60)
    print("BENCHMARKS COMPLETADOS")
    print("="*60)
    
    return results

def calculate_speedup(cpu_times, gpu_times):
    """
    Calcula el speedup de GPU respecto a CPU
    
    Args:
        cpu_times: Lista de tiempos de CPU
        gpu_times: Lista de tiempos de GPU
    
    Returns:
        list: Lista de speedups
    """
    speedups = []
    for cpu_t, gpu_t in zip(cpu_times, gpu_times):
        if cpu_t is not None and gpu_t is not None and gpu_t > 0:
            speedups.append(cpu_t / gpu_t)
        else:
            speedups.append(None)
    return speedups

def plot_execution_times(n_sizes, results):
    """
    Genera gráfico de tiempos de ejecución
    
    Args:
        n_sizes: Lista de tamaños de matriz
        results: Diccionario con resultados de cada algoritmo
    """
    plt.figure(figsize=(12, 7))
    
    colors = {1: 'blue', 2: 'red', 3: 'green'}
    markers = {1: 'o', 2: 's', 3: '^'}
    
    for alg, name in ALGORITHMS.items():
        times = results[alg]
        # Filtrar valores None
        valid_indices = [i for i, t in enumerate(times) if t is not None]
        valid_n = [n_sizes[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]
        
        plt.plot(valid_n, valid_times, 
                 label=name, 
                 marker=markers[alg], 
                 color=colors[alg],
                 linewidth=2, 
                 markersize=8)
    
    plt.xlabel('Tamaño de Matriz (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12, fontweight='bold')
    plt.title('Comparación de Tiempos de Ejecución\nMultiplicación de Matrices (N×N)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig('grafico_tiempos.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: grafico_tiempos.png")
    plt.close()

def plot_speedup(n_sizes, results):
    """
    Genera gráfico de speedup respecto a CPU
    
    Args:
        n_sizes: Lista de tamaños de matriz
        results: Diccionario con resultados de cada algoritmo
    """
    cpu_times = results[1]
    
    plt.figure(figsize=(12, 7))
    
    colors = {2: 'red', 3: 'green'}
    markers = {2: 's', 3: '^'}
    labels = {2: 'GPU Global Memory', 3: 'GPU Shared Memory'}
    
    for alg in [2, 3]:
        speedups = calculate_speedup(cpu_times, results[alg])
        
        # Filtrar valores None
        valid_indices = [i for i, s in enumerate(speedups) if s is not None]
        valid_n = [n_sizes[i] for i in valid_indices]
        valid_speedups = [speedups[i] for i in valid_indices]
        
        plt.plot(valid_n, valid_speedups,
                 label=labels[alg],
                 marker=markers[alg],
                 color=colors[alg],
                 linewidth=2,
                 markersize=8)
    
    # Línea de referencia (speedup = 1)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (CPU)')
    
    plt.xlabel('Tamaño de Matriz (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup sobre CPU', fontsize=12, fontweight='bold')
    plt.title('Aceleración de GPU respecto a CPU\n(Speedup = Tiempo_CPU / Tiempo_GPU)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Guardar el gráfico
    plt.savefig('grafico_speedup.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: grafico_speedup.png")
    plt.close()

def print_summary_table(n_sizes, results):
    """
    Imprime una tabla resumen de resultados
    
    Args:
        n_sizes: Lista de tamaños de matriz
        results: Diccionario con resultados
    """
    print("\n" + "="*80)
    print("TABLA RESUMEN DE RESULTADOS")
    print("="*80)
    print(f"{'N':<8} {'CPU (s)':<12} {'GPU Global (s)':<16} {'GPU Shared (s)':<16} {'Speedup Global':<16} {'Speedup Shared':<16}")
    print("-"*80)
    
    cpu_times = results[1]
    gpu_global_times = results[2]
    gpu_shared_times = results[3]
    
    speedup_global = calculate_speedup(cpu_times, gpu_global_times)
    speedup_shared = calculate_speedup(cpu_times, gpu_shared_times)
    
    for i, n in enumerate(n_sizes):
        cpu_str = f"{cpu_times[i]:.6f}" if cpu_times[i] is not None else "N/A"
        gpu_g_str = f"{gpu_global_times[i]:.6f}" if gpu_global_times[i] is not None else "N/A"
        gpu_s_str = f"{gpu_shared_times[i]:.6f}" if gpu_shared_times[i] is not None else "N/A"
        sp_g_str = f"{speedup_global[i]:.2f}x" if speedup_global[i] is not None else "N/A"
        sp_s_str = f"{speedup_shared[i]:.2f}x" if speedup_shared[i] is not None else "N/A"
        
        print(f"{n:<8} {cpu_str:<12} {gpu_g_str:<16} {gpu_s_str:<16} {sp_g_str:<16} {sp_s_str:<16}")
    
    print("="*80)

def main():
    """Función principal"""
    print("\n" + "="*60)
    print("SCRIPT DE BENCHMARKING Y VISUALIZACIÓN")
    print("Multiplicación de Matrices - CUDA vs OpenMP")
    print("="*60 + "\n")
    
    # Verificar que existe el ejecutable
    check_executable()
    
    # Recolectar datos
    results = collect_data()
    
    # Imprimir tabla resumen
    print_summary_table(N_SIZES, results)
    
    # Generar gráficos
    print("\nGenerando gráficos...")
    plot_execution_times(N_SIZES, results)
    plot_speedup(N_SIZES, results)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("\nArchivos generados:")
    print("  - grafico_tiempos.png")
    print("  - grafico_speedup.png")
    print("\n")

if __name__ == "__main__":
    main()

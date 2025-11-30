import os
import dclab
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def encontrar_archivos_rtdc(carpeta_base):
    """Encuentra todos los archivos .rtdc en una estructura de carpetas"""
    archivos_rtdc = []
    for root, dirs, files in os.walk(carpeta_base):
        for file in files:
            if file.endswith('.rtdc'):
                archivos_rtdc.append(os.path.join(root, file))
    return archivos_rtdc


def cargar_datos_rtdc(ruta_archivo):
    """Carga datos de un archivo .rtdc y extrae solo las características disponibles"""
    try:
        dataset = dclab.new_dataset(ruta_archivo)

        # Filtrar datos como en el artículo (área entre 25-200 μm²)
        mask = (dataset['area_um'] > 25) & (dataset['area_um'] < 200)

        # Crear diccionario de salida
        datos = {}

        # LISTA DE FEATURES POSIBLES
        features = {
            'area': 'area_um',
            'deformacion': 'deform',
            'aspect_ratio': 'aspect',
            'circularidad': 'circ',
            'inercia_cvx': 'inert_ratio_cvx',
            'inercia_raw': 'inert_ratio_raw',
        }

        # Guardar solo las features que EXISTEN
        for nombre, key in features.items():
            if key in dataset:
                datos[nombre] = dataset[key][mask]

        # Agregar metadata útil
        datos['celulas_totales'] = len(dataset['area_um'])
        datos['celulas_filtradas'] = np.sum(mask)

        return datos

    except Exception as e:
        print(f"Error cargando {ruta_archivo}: {e}")
        return None


def realizar_ks_test_completo(archivos_sanos, archivos_mds, caracteristica='area'):
    """
    Realiza KS Test Two-Sample entre grupos SANO y MDS
    """
    print(f"\n{'=' * 60}")
    print(f"KOLMOGOROV-SMIRNOV TEST - {caracteristica.upper()}")
    print(f"{'=' * 60}")

    # Cargar todos los datos
    print("Cargando datos SANOS...")
    datos_sanos = []
    for archivo in tqdm(archivos_sanos):
        datos = cargar_datos_rtdc(archivo)
        if datos is not None and caracteristica in datos and len(datos[caracteristica]) > 0:
            datos_sanos.extend(datos[caracteristica])

    print("Cargando datos MDS...")
    datos_mds = []
    for archivo in tqdm(archivos_mds):
        datos = cargar_datos_rtdc(archivo)
        if datos is not None and caracteristica in datos and len(datos[caracteristica]) > 0:
            datos_mds.extend(datos[caracteristica])

    # Convertir a arrays numpy
    datos_sanos = np.array(datos_sanos)
    datos_mds = np.array(datos_mds)

    print(f"\nESTADÍSTICAS DE MUESTRAS:")
    print(f"SANOS: {len(datos_sanos)} células")
    print(f"MDS: {len(datos_mds)} células")

    if len(datos_sanos) == 0 or len(datos_mds) == 0:
        print("ERROR: No hay suficientes datos para realizar el test")
        return None, None, None, None

    # Two-Sample KS Test
    ks_statistic, p_value = stats.ks_2samp(datos_sanos, datos_mds)

    # Resultados
    print(f"\nRESULTADOS KS TEST:")
    print(f"Estadístico KS: {ks_statistic:.6f}")
    print(f"P-value: {p_value:.6e}")
    print(f"Significativo (p < 0.05): {p_value < 0.05}")
    print(f"Significativo (p < 0.01): {p_value < 0.01}")
    print(f"Significativo (p < 0.001): {p_value < 0.001}")

    # Interpretación
    print(f"\nINTERPRETACIÓN:")
    if p_value < 0.05:
        print("✓ LAS DISTRIBUCIONES SON SIGNIFICATIVAMENTE DIFERENTES")
        print("  Hay evidencia estadística de que MDS altera la distribución celular")
    else:
        print("✗ No hay evidencia suficiente de diferencias en las distribuciones")

    return ks_statistic, p_value, datos_sanos, datos_mds


def graficar_distribuciones(datos_sanos, datos_mds, caracteristica='area'):
    """Grafica las distribuciones comparativas"""
    plt.figure(figsize=(12, 8))

    # Histograma
    plt.subplot(2, 2, 1)
    plt.hist(datos_sanos, bins=50, alpha=0.7, label='SANOS', color='blue', density=True)
    plt.hist(datos_mds, bins=50, alpha=0.7, label='MDS', color='red', density=True)

    if caracteristica == 'area':
        plt.xlabel('Área (μm²)')
    elif caracteristica == 'deformacion':
        plt.xlabel('Deformación')
    elif caracteristica == 'aspect_ratio':
        plt.xlabel('Relación de Aspecto')
    else:
        plt.xlabel(caracteristica.upper())

    plt.ylabel('Densidad')
    plt.title(f'Distribución de {caracteristica.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Boxplot
    plt.subplot(2, 2, 2)
    datos_combinados = [datos_sanos, datos_mds]
    etiquetas = ['SANOS', 'MDS']
    plt.boxplot(datos_combinados, labels=etiquetas)
    plt.title(f'Boxplot - {caracteristica.upper()}')
    plt.grid(True, alpha=0.3)

    # ECDF (Empirical Cumulative Distribution Function)
    plt.subplot(2, 2, 3)
    # Para SANOS
    x_sano = np.sort(datos_sanos)
    y_sano = np.arange(1, len(x_sano) + 1) / len(x_sano)
    plt.plot(x_sano, y_sano, label='SANOS', color='blue', linewidth=2)

    # Para MDS
    x_mds = np.sort(datos_mds)
    y_mds = np.arange(1, len(x_mds) + 1) / len(x_mds)
    plt.plot(x_mds, y_mds, label='MDS', color='red', linewidth=2)

    if caracteristica == 'area':
        plt.xlabel('Área (μm²)')
    else:
        plt.xlabel(caracteristica.upper())

    plt.ylabel('Probabilidad Acumulada')
    plt.title('Función de Distribución Acumulada (ECDF)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Estadísticas descriptivas
    plt.subplot(2, 2, 4)
    # 1. Calcular las estadísticas (todavía como floats o arrays)
    stats_sanos = {
        'N': len(datos_sanos),
        'Media': np.mean(datos_sanos),
        'Mediana': np.median(datos_sanos),
        'Std': np.std(datos_sanos),
        'MAD': stats.median_abs_deviation(datos_sanos)
    }

    stats_mds = {
        'N': len(datos_mds),
        'Media': np.mean(datos_mds),
        'Mediana': np.median(datos_mds),
        'Std': np.std(datos_mds),
        'MAD': stats.median_abs_deviation(datos_mds)
    }

    # 2. Definir las etiquetas de columna y de fila (Grupo)
    columnas = ['N', 'Media', 'Mediana', 'Std', 'MAD']
    etiquetas_columna = ['N', 'Media', 'Mediana', 'Std', 'MAD']

    # 3. Crear el cellText como una lista de listas de STRINGS formateados
    cellText = [
        # Fila SANOS
        ['SANOS'] + [f"{stats_sanos[col]:.4f}" if col != 'N' else str(stats_sanos[col]) for col in columnas],
        # Fila MDS
        ['MDS'] + [f"{stats_mds[col]:.4f}" if col != 'N' else str(stats_mds[col]) for col in columnas],
    ]

    # 4. Ajustar colLabels para incluir 'Grupo'
    etiquetas_finales = ['Grupo'] + etiquetas_columna

    plt.axis('off')
    tabla = plt.table(cellText=cellText,
                      colLabels=etiquetas_finales,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.5)
    plt.title('Estadísticas Descriptivas')

    plt.tight_layout()
    plt.show()


def analisis_completo(carpeta_base):
    """Análisis completo de todos los datos"""
    print("BÚSQUEDA DE ARCHIVOS .rtdc...")
    todos_archivos = encontrar_archivos_rtdc(carpeta_base)

    print(f"Se encontraron {len(todos_archivos)} archivos .rtdc")

    # Clasificar archivos
    archivos_sanos = [f for f in todos_archivos if "CD34_healthy" in f]
    archivos_mds = [f for f in todos_archivos if "MDS" in f and "healthy" not in f]

    print(f"\nCLASIFICACIÓN:")
    print(f"Archivos SANOS: {len(archivos_sanos)}")
    print(f"Archivos MDS: {len(archivos_mds)}")

    # Mostrar archivos encontrados
    print(f"\nARCHIVOS SANOS:")
    for archivo in archivos_sanos:
        print(f"  - {os.path.basename(archivo)}")

    print(f"\nARCHIVOS MDS (primeros 10):")
    for archivo in archivos_mds[:10]:
        print(f"  - {os.path.basename(archivo)}")
    if len(archivos_mds) > 10:
        print(f"  ... y {len(archivos_mds) - 10} más")

    # Características a analizar basadas en tu output
    caracteristicas = [
        'area',  # Área celular - LA MÁS IMPORTANTE según el artículo
        'deformacion',  # Deformación - propiedades mecánicas
        'aspect_ratio',  # Relación de aspecto
        'circularidad',  # Circularidad
        'inercia_cvx',  # Relación de inercia (convex hull)
        'brillo_avg'  # Brillo promedio
    ]

    resultados = {}
    for caracteristica in caracteristicas:
        ks_stat, p_val, datos_sanos, datos_mds = realizar_ks_test_completo(
            archivos_sanos, archivos_mds, caracteristica
        )

        if ks_stat is not None:
            resultados[caracteristica] = {
                'ks_statistic': ks_stat,
                'p_value': p_val,
                'datos_sanos': datos_sanos,
                'datos_mds': datos_mds
            }

            # Graficar resultados
            graficar_distribuciones(datos_sanos, datos_mds, caracteristica)

    # Resumen final
    print(f"\n{'=' * 60}")
    print("RESUMEN FINAL DE RESULTADOS")
    print(f"{'='
             '' * 60}")

    for carac, res in resultados.items():
        sign = "✓" if res['p_value'] < 0.05 else "✗"
        print(f"{carac.upper():<15} | KS: {res['ks_statistic']:.4f} | p-value: {res['p_value']:.2e} {sign}")


# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Configurar tu ruta base aquí
    CARPETA_BASE = r"D:\Documentos\Programacion\Parcial 3\01_ExperimentalData"

    # Verificar si la carpeta existe
    if not os.path.exists(CARPETA_BASE):
        print(f"ERROR: No se encuentra la carpeta {CARPETA_BASE}")
        print("Por favor, verifica la ruta")
    else:
        print("INICIANDO ANÁLISIS KS TEST MDS vs SANO")
        print(f"Directorio base: {CARPETA_BASE}")

        # Ejecutar análisis completo
        analisis_completo(CARPETA_BASE)
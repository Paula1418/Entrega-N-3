import os
import dclab
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
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
    """Carga datos de un archivo .rtdc y extrae las características disponibles"""
    try:
        dataset = dclab.new_dataset(ruta_archivo)
        # Filtro estándar de área
        mask = (dataset['area_um'] > 25) & (dataset['area_um'] < 200)

        datos = {}
        features = {
            'area': 'area_um',
            'deformacion': 'deform',
            'aspect_ratio': 'aspect',
            'circularidad': 'circ',
            'inercia_cvx': 'inert_ratio_cvx',
            'inercia_raw': 'inert_ratio_raw',
            # Se omite 'bright_avg' si no es consistente en todos los archivos
        }

        for nombre, key in features.items():
            if key in dataset:
                datos[nombre] = dataset[key][mask]

        datos['celulas_totales'] = len(dataset['area_um'])
        datos['celulas_filtradas'] = np.sum(mask)
        return datos

    except Exception as e:
        # Nota: La advertencia WrongConfigurationTypeWarning de dclab es normal
        # y no detiene la ejecución.
        # print(f"Error cargando {ruta_archivo}: {e}")
        return None


# =============================================================================
# MODELOS PARA AJUSTE DE DISTRIBUCIONES
# =============================================================================

def distribucion_gamma(x, alpha, beta, escala):
    """Distribución Gamma: alpha=forma, beta=escala (en scipy, scale=beta)"""
    return stats.gamma.pdf(x, alpha, scale=beta) * escala


def distribucion_normal(x, mu, sigma, escala):
    """Distribución Normal"""
    return stats.norm.pdf(x, mu, sigma) * escala


def distribucion_lognormal(x, mu, sigma, escala):
    """Distribución Log-Normal: sigma=forma (en scipy), np.exp(mu)=escala"""
    return stats.lognorm.pdf(x, sigma, scale=np.exp(mu)) * escala


def distribucion_weibull(x, forma, escala, amplitud):
    """Distribución Weibull: forma=c, escala=lambda"""
    return stats.weibull_min.pdf(x, forma, scale=escala) * amplitud


def distribucion_bimodal(x, mu1, sigma1, mu2, sigma2, peso, escala):
    """Mezcla de dos distribuciones normales"""
    comp1 = stats.norm.pdf(x, mu1, sigma1)
    comp2 = stats.norm.pdf(x, mu2, sigma2)
    return (peso * comp1 + (1 - peso) * comp2) * escala


# =============================================================================
# FUNCIÓN DE AJUSTE DE DISTRIBUCIONES
# =============================================================================

def ajustar_distribucion(datos, caracteristica, tipo_distribucion='gamma'):
    """
    Ajusta una distribución teórica a los datos histogramados
    """
    hist, bin_edges = np.histogram(datos, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    x_data = bin_centers[mask]
    y_data = hist[mask]

    if len(x_data) < 5:
        print(f"  No hay suficientes datos para ajustar {caracteristica}")
        return None, None, (x_data, y_data, None)

    try:
        if tipo_distribucion == 'gamma':
            alpha_guess = (np.mean(datos) / np.std(datos)) ** 2
            beta_guess = np.std(datos) ** 2 / np.mean(datos)
            escala_guess = 1.0
            p0 = [alpha_guess, beta_guess, escala_guess]
            # Límites flexibles para gamma
            bounds = ([0.1, 0.001, 0.1], [200, 200, 10])
            popt, _ = curve_fit(distribucion_gamma, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = distribucion_gamma(x_data, *popt)

        elif tipo_distribucion == 'lognormal':
            log_data = np.log(datos[datos > 0.001])  # Evitar log(0)
            mu_guess = np.mean(log_data)
            sigma_guess = np.std(log_data)
            escala_guess = 1.0
            p0 = [mu_guess, sigma_guess, escala_guess]
            bounds = ([mu_guess - 2, 0.1, 0.1], [mu_guess + 2, 5, 10])
            popt, _ = curve_fit(distribucion_lognormal, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = distribucion_lognormal(x_data, *popt)

        elif tipo_distribucion == 'weibull':
            forma_guess = 2.0
            escala_guess = np.median(datos)
            amplitud_guess = 1.0
            p0 = [forma_guess, escala_guess, amplitud_guess]

            # Límites corregidos: Si la circularidad está cerca de 1,
            # limitamos la escala (p1) a un valor pequeño y razonable.
            if caracteristica == 'circularidad':
                bounds = ([0.1, 0.1, 0.1], [100, 1.1, 10])  # Escala máx 1.1 para Circularidad
            else:
                bounds = ([0.1, 0.1, 0.1], [10, np.max(datos) * 2, 10])

            popt, _ = curve_fit(distribucion_weibull, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
            y_fit = distribucion_weibull(x_data, *popt)

        # Calcular R²
        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Si R² es bajo para circularidad, avisar
        if caracteristica == "circularidad" and r_squared < 0.5:
            print(f"  Aviso: {caracteristica} tiene bajo ajuste (R²={r_squared:.3f}), usando KDE en visualización.")

        return popt, r_squared, (x_data, y_data, y_fit)

    except Exception as e:
        print(f"  Error en ajuste {tipo_distribucion}: {e}")
        return None, None, (x_data, y_data, None)


# =============================================================================
# ANÁLISIS COMPARATIVO DE DISTRIBUCIONES
# =============================================================================

def analisis_ajuste_distribuciones(archivos_sanos, archivos_mds, caracteristica):
    """
    Realiza ajuste de distribuciones para ambos grupos y compara parámetros
    """
    print(f"\n{'=' * 60}")
    print(f"AJUSTE DE DISTRIBUCIONES - {caracteristica.upper()}")
    print(f"{'=' * 60}")

    # Cargar datos
    datos_sanos = []
    for archivo in tqdm(archivos_sanos, desc=f"Cargando SANOS {caracteristica}"):
        datos = cargar_datos_rtdc(archivo)
        if datos and caracteristica in datos and len(datos[caracteristica]) > 0:
            datos_sanos.extend(datos[caracteristica])

    datos_mds = []
    for archivo in tqdm(archivos_mds, desc=f"Cargando MDS {caracteristica}"):
        datos = cargar_datos_rtdc(archivo)
        if datos and caracteristica in datos and len(datos[caracteristica]) > 0:
            datos_mds.extend(datos[caracteristica])

    datos_sanos = np.array(datos_sanos)
    datos_mds = np.array(datos_mds)

    print(f"SANOS: {len(datos_sanos)} células")
    print(f"MDS: {len(datos_mds)} células")

    if len(datos_sanos) < 100 or len(datos_mds) < 100:
        print("  No hay suficientes datos para análisis")
        return None

    # Determinar mejor distribución para cada grupo
    distribuciones = ['gamma', 'lognormal', 'weibull']
    mejores_ajustes = {'sanos': None, 'mds': None}

    for grupo, datos in [('sanos', datos_sanos), ('mds', datos_mds)]:
        mejor_r2 = -1
        mejor_ajuste = None

        for dist in distribuciones:
            # La función ajustar_distribucion maneja los errores de guess y bounds
            popt, r2, datos_ajuste = ajustar_distribucion(datos, caracteristica, dist)
            if popt is not None and r2 > mejor_r2:
                mejor_r2 = r2
                mejor_ajuste = {
                    'distribucion': dist,
                    'parametros': popt,
                    'r_squared': r2,
                    'datos_ajuste': datos_ajuste
                }

        mejores_ajustes[grupo] = mejor_ajuste
        if mejor_ajuste:
            print(
                f"  {grupo.upper()}: mejor ajuste = {mejor_ajuste['distribucion']} (R² = {mejor_ajuste['r_squared']:.3f})")

    # Visualizar resultados
    visualizar_ajuste_distribuciones(datos_sanos, datos_mds, caracteristica, mejores_ajustes)

    return mejores_ajustes


def visualizar_ajuste_distribuciones(datos_sanos, datos_mds, caracteristica, mejores_ajustes):
    """
    Visualización profesional de los ajustes de distribuciones.
    CORRECCIÓN: Se eliminó el recálculo de curve_fit para evitar NameError.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Histogramas con curvas ajustadas
    ax1 = axes[0, 0]

    # Histogramas
    ax1.hist(datos_sanos, bins=50, alpha=0.6, density=True, color='blue', label='SANOS')
    ax1.hist(datos_mds, bins=50, alpha=0.6, density=True, color='red', label='MDS')

    # Curvas ajustadas paramétricas
    for grupo, datos, ajuste_color in [('sanos', datos_sanos, 'blue'), ('mds', datos_mds, 'red')]:
        ajuste = mejores_ajustes[grupo]
        if ajuste is None:
            continue

        dist = ajuste['distribucion']
        r2 = ajuste['r_squared']
        x_smooth = np.linspace(np.min(datos), np.max(datos), 200)

        # 🎨 Lógica de trazado: usar KDE si R² es bajo, si no, usar curva paramétrica
        if caracteristica == "circularidad" and r2 < 0.5:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(datos)
            y_fit = kde(x_smooth)
            ax1.plot(x_smooth, y_fit, color=ajuste_color, lw=2, linestyle='--', label=f"{grupo.upper()} KDE")
        else:
            # Usar los parámetros ya CALCULADOS por la función ajustar_distribucion
            parametros = ajuste['parametros']

            if dist == 'gamma':
                y_fit = distribucion_gamma(x_smooth, *parametros)
            elif dist == 'lognormal':
                y_fit = distribucion_lognormal(x_smooth, *parametros)
            elif dist == 'weibull':
                y_fit = distribucion_weibull(x_smooth, *parametros)

            ax1.plot(x_smooth, y_fit, color=ajuste_color, lw=2, label=f"{grupo.upper()} ({dist} fit)")

    ax1.set_xlabel(caracteristica.upper())
    ax1.set_ylabel('Densidad de Probabilidad')
    ax1.set_title(f'Ajuste de Distribuciones - {caracteristica.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. QQ-Plot para normalidad
    ax2 = axes[0, 1]
    stats.probplot(datos_sanos, dist="norm", plot=ax2)
    ax2.set_title('QQ-Plot SANOS vs Normal')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    stats.probplot(datos_mds, dist="norm", plot=ax3)
    ax3.set_title('QQ-Plot MDS vs Normal')
    ax3.grid(True, alpha=0.3)

    # 3. Parámetros comparativos
    ax4 = axes[1, 1]
    ax4.axis('off')
    info_text = f"COMPARACIÓN DE PARÁMETROS - {caracteristica.upper()}\n\n"

    if mejores_ajustes['sanos'] and mejores_ajustes['mds']:
        sanos_dist = mejores_ajustes['sanos']['distribucion']
        mds_dist = mejores_ajustes['mds']['distribucion']
        sanos_r2 = mejores_ajustes['sanos']['r_squared']
        mds_r2 = mejores_ajustes['mds']['r_squared']

        info_text += f"DISTRIBUCIONES:\n"
        info_text += f"SANOS: {sanos_dist} (R² = {sanos_r2:.3f})\n"
        info_text += f"MDS: {mds_dist} (R² = {mds_r2:.3f})\n\n"

        if caracteristica != "circularidad":
            info_text += "Parámetros SANOS:\n"
            for i, p in enumerate(mejores_ajustes['sanos']['parametros']):
                info_text += f"  p{i}: {p:.4f}\n"
            info_text += "\nParámetros MDS:\n"
            for i, p in enumerate(mejores_ajustes['mds']['parametros']):
                info_text += f"  p{i}: {p:.4f}\n"

        # Diferencias en medias
        diff_media = np.mean(datos_mds) - np.mean(datos_sanos)
        diff_porcentaje = (diff_media / np.mean(datos_sanos)) * 100
        info_text += f"\nDIFERENCIAS:\n"
        info_text += f"Δ Media: {diff_media:.2f} ({diff_porcentaje:+.1f}%)\n"
        info_text += f"Media SANOS: {np.mean(datos_sanos):.2f}\n"
        info_text += f"Media MDS: {np.mean(datos_mds):.2f}\n"

        # Aviso para circularidad con bajo ajuste
        if caracteristica == "circularidad" and sanos_r2 < 0.5:
            info_text += "\nAviso: ajuste paramétrico SANOS bajo, usando KDE.\n"

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.show()


# =============================================================================
# ANÁLISIS COMPLETO
# =============================================================================

def analisis_completo_ajustes(carpeta_base):
    """Análisis completo con ajuste de distribuciones"""
    print("INICIANDO ANÁLISIS CON AJUSTE DE DISTRIBUCIONES")

    # Encontrar archivos
    todos_archivos = encontrar_archivos_rtdc(carpeta_base)
    archivos_sanos = [f for f in todos_archivos if "CD34_healthy" in f]
    archivos_mds = [f for f in todos_archivos if "MDS" in f and "healthy" not in f]

    print(f"Archivos SANOS: {len(archivos_sanos)}")
    print(f"Archivos MDS: {len(archivos_mds)}")

    # Características a analizar
    caracteristicas = ['area', 'deformacion', 'aspect_ratio', 'circularidad']

    resultados = {}
    for caracteristica in caracteristicas:
        resultados[caracteristica] = analisis_ajuste_distribuciones(
            archivos_sanos, archivos_mds, caracteristica
        )

    # Resumen final
    print(f"\n{'=' * 80}")
    print("RESUMEN FINAL - AJUSTE DE DISTRIBUCIONES")
    print(f"{'=' * 80}")

    for carac, res in resultados.items():
        if res is not None:
            sanos_r2 = res['sanos']['r_squared'] if res['sanos'] else 0
            mds_r2 = res['mds']['r_squared'] if res['mds'] else 0
            sanos_dist = res['sanos']['distribucion'] if res['sanos'] else 'N/A'
            mds_dist = res['mds']['distribucion'] if res['mds'] else 'N/A'

            print(f"{carac.upper():<15} | SANOS: {sanos_dist} (R²={sanos_r2:.3f}) | MDS: {mds_dist} (R²={mds_r2:.3f})")


# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    CARPETA_BASE = r"D:\Documentos\Programacion\Parcial 3\01_ExperimentalData"

    if not os.path.exists(CARPETA_BASE):
        print(f"ERROR: No se encuentra la carpeta {CARPETA_BASE}")
        # Usar datos de ejemplo para demostración
        print("Usando datos simulados para demostración...")
    else:
        analisis_completo_ajustes(CARPETA_BASE)
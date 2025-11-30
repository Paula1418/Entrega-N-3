import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Cargar datos
df = pd.read_csv(r"D:\Documentos\Programacion\Parcial 3\GBM-PRJNA482620.Response.csv")

print("Columnas del dataset:", df.columns.tolist())
print("Primeras filas:")
print(df.head())

# Seleccionar los 8 genes de interés
genes_interes = ['A2M', 'ABCB1', 'ABCG2', 'ABCC1', 'AATK', 'ABAT', 'ABL1', 'AAMP']

# Verificar qué genes están disponibles
genes_disponibles = [gen for gen in genes_interes if gen in df['GENE_SYMBOL'].values]
print(f"Genes disponibles: {genes_disponibles}")


# DEFINIR LAS FUNCIONES DEL MODELO PRIMERO
def modelo_acoplado_gen_tumor(t, y, parametros_base, expresion_paciente):
    """
    Sistema dinámico donde la expresión génica modula parámetros en tiempo real
    """
    T, E = y  # Tumor, Células inmunes

    # Calcular parámetros instantáneos basados en expresión génica
    g = expresion_paciente
    a = parametros_base['a0'] * (1 + 0.3 * g.get('ABCB1', 0.5) - 0.2 * g.get('AATK', 0.5))
    c = parametros_base['c0'] * (1 - 0.4 * g.get('A2M', 0.5))
    delta = parametros_base['delta0'] * (1 + 0.5 * g.get('ABCG2', 0.5))

    # Sistema de ecuaciones
    dTdt = a * T * (1 - T / parametros_base['K0']) - c * E * T
    dEdt = (parametros_base['sigma'] +
            (parametros_base['rho'] * E * T) / (parametros_base['eta'] + T) -
            delta * E * T - parametros_base['mu'] * E)

    return [dTdt, dEdt]


def calcular_parametros_paciente(expresion_paciente, parametros_base):
    """
    Calcular parámetros del modelo basado en expresión génica de un paciente
    """
    g = expresion_paciente

    # Promedio de genes pro-crecimiento (solo los disponibles)
    genes_pro_crecimiento = []
    for gen in ['ABCB1', 'ABCG2', 'ABL1', 'AAMP']:
        if gen in g:
            genes_pro_crecimiento.append(g[gen])

    pro_crecimiento = np.mean(genes_pro_crecimiento) if genes_pro_crecimiento else 0.5

    # Cálculo de parámetros (usar valores por defecto si el gen no está disponible)
    a = parametros_base['a0'] * (1 + 0.5 * pro_crecimiento - 0.4 * g.get('AATK', 0.5))
    c = parametros_base['c0'] * (1 - 0.3 * g.get('A2M', 0.5) - 0.2 * g.get('ABCC1', 0.5))
    delta = parametros_base['delta0'] * (1 + 0.4 * g.get('A2M', 0.5))
    K = parametros_base['K0'] * (1 + 0.6 * g.get('AAMP', 0.5))

    return {
        'a': max(a, 0.1),  # Evitar valores negativos
        'K': K,
        'c': max(c, 0.01),
        'sigma': parametros_base['sigma'],
        'rho': parametros_base['rho'],
        'eta': parametros_base['eta'],
        'delta': delta,
        'mu': parametros_base['mu']
    }


def simular_paciente_completo(expresion_paciente, parametros_base, t_span=(0, 50), y0=[1e4, 1e3]):
    """
    Simular dinámica tumoral completa para un paciente específico
    """
    # Resolver sistema acoplado
    sol = solve_ivp(modelo_acoplado_gen_tumor, t_span, y0,
                    args=(parametros_base, expresion_paciente),
                    t_eval=np.linspace(t_span[0], t_span[1], 500),
                    method='RK45')

    # Calcular parámetros para análisis
    parametros = calcular_parametros_paciente(expresion_paciente, parametros_base)

    return sol, parametros


# Parámetros base del modelo
parametros_base = {
    'a0': 0.8,  # Tasa crecimiento tumoral base
    'K0': 1e6,  # Capacidad de carga base
    'c0': 0.05,  # Tasa killing inmune base
    'sigma': 0.1,  # Influx constante células inmunes
    'rho': 0.8,  # Tasa reclutamiento estimulado
    'eta': 1e5,  # Constante semi-saturación
    'delta0': 5e-7,  # Tasa inhibición inmune base
    'mu': 0.2  # Tasa muerte células inmunes
}

# CONTINUAR CON EL RESTO DEL CÓDIGO SOLO SI HAY SUFICIENTES GENES
if len(genes_disponibles) < 3:
    print("ERROR: No hay suficientes genes disponibles")
    print("Algunos genes en el dataset:", df['GENE_SYMBOL'].head(20).tolist())
else:
    # Extraer datos de expresión para estos genes
    expresion_genes = df[df['GENE_SYMBOL'].isin(genes_disponibles)].set_index('GENE_SYMBOL')

    # Las columnas de muestras
    muestras = expresion_genes.columns.tolist()
    print(f"Muestras disponibles: {len(muestras)}")
    print(f"Primeras 5 muestras: {muestras[:5]}")

    # Normalizar expresión entre 0 y 1 para cada gen
    expresion_normalizada = expresion_genes.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        axis=1
    )

    print(f"\nExpresión normalizada shape: {expresion_normalizada.shape}")
    print(expresion_normalizada.head())

    # Seleccionar pacientes para comparar
    pacientes_comparar = expresion_normalizada.columns[:3].tolist()
    print(f"\nPacientes a simular: {pacientes_comparar}")

    # Configurar gráficos
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)

    # =========================================================================
    # 1. DINÁMICA TEMPORAL ACOPLADA (MODELO PRINCIPAL)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    t_eval = np.linspace(0, 50, 500)

    for i, paciente in enumerate(pacientes_comparar):
        expresion_pac = expresion_normalizada[paciente].to_dict()

        # Resolver sistema acoplado
        sol, params = simular_paciente_completo(expresion_pac, parametros_base)

        T = np.maximum(sol.y[0], 0)
        E = np.maximum(sol.y[1], 0)

        ax1.plot(sol.t, T, label=f'Paciente {paciente} - Tumor',
                 color=f'C{i}', linewidth=2.5)
        ax1.plot(sol.t, E, '--', label=f'Paciente {paciente} - Inmune',
                 color=f'C{i}', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Tiempo (días)', fontsize=12)
    ax1.set_ylabel('Densidad Celular', fontsize=12)
    ax1.set_title('Dinámica Temporal Tumor-Inmune Acoplada a Expresión Génica', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1)

    # =========================================================================
    # 2. DIAGRAMAS DE FASE
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    for i, (ax, paciente) in enumerate(zip([ax2, ax3, ax4], pacientes_comparar)):
        expresion_pac = expresion_normalizada[paciente].to_dict()
        sol, params = simular_paciente_completo(expresion_pac, parametros_base)

        T = np.maximum(sol.y[0], 0)
        E = np.maximum(sol.y[1], 0)

        ax.plot(T, E, color=f'C{i}', linewidth=2.5)
        ax.scatter(T[0], E[0], color='green', s=80, label='Inicio', zorder=5, edgecolors='black')
        ax.scatter(T[-1], E[-1], color='red', s=80, label='Final', zorder=5, edgecolors='black')

        ax.set_xlabel('Células Tumorales (T)')
        ax.set_ylabel('Células Inmunes (E)')
        ax.set_title(f'Diagrama Fase - {paciente}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # =========================================================================
    # 3. HEATMAP DE EXPRESIÓN GÉNICA
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    expresion_plot = expresion_normalizada[pacientes_comparar]

    # Reordenar genes para mejor visualización
    genes_ordenados = sorted(genes_disponibles,
                             key=lambda x: expresion_plot.loc[x].mean(),
                             reverse=True)
    expresion_plot = expresion_plot.loc[genes_ordenados]

    im = ax5.imshow(expresion_plot.values, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    ax5.set_xticks(range(len(pacientes_comparar)))
    ax5.set_xticklabels(pacientes_comparar, rotation=45, ha='right')
    ax5.set_yticks(range(len(genes_ordenados)))
    ax5.set_yticklabels(genes_ordenados)
    ax5.set_title('Expresión Génica Normalizada', fontweight='bold')

    # Añadir valores numéricos al heatmap
    for i in range(len(genes_ordenados)):
        for j in range(len(pacientes_comparar)):
            ax5.text(j, i, f'{expresion_plot.values[i, j]:.2f}',
                     ha="center", va="center",
                     color="white" if expresion_plot.values[i, j] > 0.6 else "black",
                     fontweight='bold')

    plt.colorbar(im, ax=ax5, label='Expresión Normalizada')

    # =========================================================================
    # 4. ANÁLISIS DE SENSIBILIDAD MEJORADO
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 1:])

    # Calcular tamaño tumoral final vs expresión génica (primeros 8 pacientes)
    pacientes_analizar = expresion_normalizada.columns[:min(8, len(expresion_normalizada.columns))].tolist()
    tumores_finales = []
    parametros_todos = []

    for paciente in pacientes_analizar:
        expresion_pac = expresion_normalizada[paciente].to_dict()
        sol, params = simular_paciente_completo(expresion_pac, parametros_base)
        tumor_final = np.maximum(sol.y[0, -1], 1)  # Mínimo 1 célula para log
        tumores_finales.append(tumor_final)
        parametros_todos.append(params)

    # Gráfico de sensibilidad múltiple
    genes_analizar = [gen for gen in ['A2M', 'ABCB1', 'AATK', 'ABCG2', 'ABCC1', 'AAMP'] if gen in genes_disponibles]

    # Crear subgráficos para diferentes correlaciones
    width = 0.15
    x_pos = np.arange(len(genes_analizar))

    # Correlación con tumor final
    correlaciones_tumor = []
    for gen in genes_analizar:
        expr_gen = [expresion_normalizada.loc[gen, pac] for pac in pacientes_analizar]
        correl = np.corrcoef(expr_gen, np.log10(tumores_finales))[0, 1]
        correlaciones_tumor.append(correl if not np.isnan(correl) else 0)

    # Correlación con tasa de crecimiento
    correlaciones_crecimiento = []
    for gen in genes_analizar:
        expr_gen = [expresion_normalizada.loc[gen, pac] for pac in pacientes_analizar]
        tasas = [p['a'] for p in parametros_todos]
        correl = np.corrcoef(expr_gen, tasas)[0, 1]
        correlaciones_crecimiento.append(correl if not np.isnan(correl) else 0)

    bars1 = ax6.bar(x_pos - width / 2, correlaciones_tumor, width,
                    label='Correlación con Tumor Final',
                    color=['red' if x > 0 else 'blue' for x in correlaciones_tumor],
                    alpha=0.7)

    bars2 = ax6.bar(x_pos + width / 2, correlaciones_crecimiento, width,
                    label='Correlación con Tasa Crecimiento',
                    color=['darkred' if x > 0 else 'darkblue' for x in correlaciones_crecimiento],
                    alpha=0.7)

    ax6.set_xlabel('Genes')
    ax6.set_ylabel('Coeficiente de Correlación')
    ax6.set_title('Sensibilidad: Correlación Gen-Tumor Final y Tasa de Crecimiento', fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(genes_analizar, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Añadir línea en y=0
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # ANÁLISIS NUMÉRICO DETALLADO
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANÁLISIS DETALLADO POR PACIENTE - MODELO ACOPLADO")
    print("=" * 70)

    for paciente in pacientes_comparar:
        expresion_pac = expresion_normalizada[paciente].to_dict()
        sol, params = simular_paciente_completo(expresion_pac, parametros_base)

        tumor_final = np.maximum(sol.y[0, -1], 0)
        inmune_final = np.maximum(sol.y[1, -1], 0)

        # Calcular parámetros del modelo acoplado
        g = expresion_pac
        a_acoplado = parametros_base['a0'] * (1 + 0.3 * g.get('ABCB1', 0.5) - 0.2 * g.get('AATK', 0.5))
        c_acoplado = parametros_base['c0'] * (1 - 0.4 * g.get('A2M', 0.5))

        print(f"\n📊 Paciente: {paciente}")
        print(f"   Tumor final: {tumor_final:.2e}")
        print(f"   Células inmunes finales: {inmune_final:.2e}")
        if inmune_final > 0:
            print(f"   Ratio T/E: {tumor_final / inmune_final:.2f}")

        print(f"   Parámetros acoplados:")
        print(f"     - Tasa crecimiento (a): {a_acoplado:.3f}")
        print(f"     - Tasa killing (c): {c_acoplado:.3f}")

        # Clasificar resultado
        if tumor_final < 1e3:
            estado = "✅ ELIMINACIÓN TUMORAL"
        elif tumor_final < 1e5:
            estado = "⚠️  EQUILIBRIO INMUNE"
        else:
            estado = "❌ ESCAPE TUMORAL"

        print(f"   Estado: {estado}")

    # Análisis de todos los pacientes
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS GLOBALES - MODELO ACOPLADO")
    print("=" * 70)

    tumores_todos = []
    for paciente in expresion_normalizada.columns[:10]:  # Solo primeros 10 para velocidad
        try:
            expresion_pac = expresion_normalizada[paciente].to_dict()
            sol, params = simular_paciente_completo(expresion_pac, parametros_base)
            tumor_final = np.maximum(sol.y[0, -1], 0)
            tumores_todos.append(tumor_final)
        except Exception as e:
            print(f"Error con paciente {paciente}: {e}")
            continue

    print(f"Pacientes simulados: {len(tumores_todos)}")
    print(f"Tumor promedio final: {np.mean(tumores_todos):.2e}")
    print(f"Desviación estándar: {np.std(tumores_todos):.2e}")

    # Clasificación de resultados
    eliminados = sum(1 for t in tumores_todos if t < 1e3)
    equilibrio = sum(1 for t in tumores_todos if 1e3 <= t < 1e5)
    escapados = sum(1 for t in tumores_todos if t >= 1e5)

    print(f"\n📈 Distribución de Resultados:")
    print(f"   Eliminación tumoral: {eliminados} pacientes ({eliminados / len(tumores_todos) * 100:.1f}%)")
    print(f"   Equilibrio inmune: {equilibrio} pacientes ({equilibrio / len(tumores_todos) * 100:.1f}%)")
    print(f"   Escape tumoral: {escapados} pacientes ({escapados / len(tumores_todos) * 100:.1f}%)")
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. Generación del factor r usando la matrícula como semilla
matricula = 2145151
random.seed(matricula)
r = random.random()

# Definición del rango de temperatura [T_min, T_max]
T_min = 5.0
T_max = 150.0

# Niveles de temperatura (Diseño Factorial de 3 niveles)
frac_bajo = 0.3 * r
frac_medio = 0.3 * r + 0.3
frac_alto = 0.3 * r + 0.6

temps = {
    "Bajo": T_min + frac_bajo * (T_max - T_min),
    "Medio": T_min + frac_medio * (T_max - T_min),
    "Alto": T_min + frac_alto * (T_max - T_min),
}

sizes = [5, 10, 15]
replicas = 10


# Funciones de soporte para el TSP y Simulated Annealing
def generate_cities(n, seed_inst):
  rng_inst = random.Random(seed_inst)
  return [
      (rng_inst.uniform(0, 10), rng_inst.uniform(0, 10)) for _ in range(n)
  ]


def tour_length(points, route):
  return sum(
      math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
      for i in range(len(route))
  )


def two_opt_move(route, i, k):
  return route[:i] + route[i : k + 1][::-1] + route[k + 1 :]


def simulated_annealing(points, T_initial, seed_run):
  rng = random.Random(seed_run)
  n = len(points)
  current_route = list(range(n))
  rng.shuffle(current_route)
  current_cost = tour_length(points, current_route)
  best_cost = current_cost
  T = T_initial
  T_min_sa = 0.01
  alpha = 0.95

  while T > T_min_sa:
    if n > 2:
      i = rng.randint(1, n - 2)
      k = rng.randint(i + 1, n - 1)
      neighbor = two_opt_move(current_route, i, k)
      neighbor_cost = tour_length(points, neighbor)
      delta = neighbor_cost - current_cost
      if delta < 0 or rng.random() < math.exp(-delta / T):
        current_route = neighbor
        current_cost = neighbor_cost
        if current_cost < best_cost:
          best_cost = current_cost
    T *= alpha
  return best_cost


# 2. Ejecución del experimento (90 corridas en total)
data = []
for n in sizes:
  points = generate_cities(n, seed_inst=42 + n)
  for temp_name, t_val in temps.items():
    for rep in range(replicas):
      cost = simulated_annealing(points, t_val, seed_run=1000 * n + rep)
      data.append({
          "temperatura_nivel": temp_name,
          "temperatura_val": t_val,
          "tamano": n,
          "replica": rep,
          "costo": cost,
      })

df = pd.DataFrame(data)

# 3. ANOVA de dos factores con interacción
modelo = ols("costo ~ C(temperatura_nivel) * C(tamano)", data=df).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print(tabla_anova)


# 4. Cálculo de Mediana Bootstrap (IC 95%)
def bootstrap_median_ci(series, n_boot=1000, alpha=0.05):
  values = series.values
  boot_medians = [
      np.median(np.random.choice(values, size=len(values), replace=True))
      for _ in range(n_boot)
  ]
  return (
      np.median(values),
      np.percentile(boot_medians, 100 * (alpha / 2)),
      np.percentile(boot_medians, 100 * (1 - alpha / 2)),
  )


summary_results = []
for n in sizes:
  for temp_name in temps.keys():
    subset = df[(df["tamano"] == n) & (df["temperatura_nivel"] == temp_name)][
        "costo"
    ]
    med, ci_low, ci_high = bootstrap_median_ci(subset)
    summary_results.append({
        "tamano": n,
        "temperatura": temp_name,
        "T_valor": temps[temp_name],
        "mediana_bootstrap": med,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
    })
# 5. Visualización limpia y profesional para el reporte en PDF
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gráfica 1: Comportamiento por Nivel de Temperatura y Tamaño (Efectos / Interacción)
mean_data = (
    df.groupby(["tamano", "temperatura_nivel"])["costo"]
    .mean()
    .unstack()[["Bajo", "Medio", "Alto"]]
)
mean_data.plot(
    kind="bar",
    ax=axes[0],
    colormap="viridis",
    edgecolor="black",
    rot=0,
    linewidth=0.8,
)
axes[0].set_title(
    "Efecto de la Temperatura Inicial en el Costo del TSP",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel("Tamaño del Problema ($n$ ciudades)", fontsize=11)
axes[0].set_ylabel("Costo Promedio (Distancia Total)", fontsize=11)
axes[0].legend(title="Nivel de Temperatura", frameon=True)
axes[0].grid(axis="y", linestyle=":", alpha=0.7)

# Gráfica 2: Distribución de Costos (Boxplot global por nivel de temperatura)
df["temperatura_nivel"] = pd.Categorical(
    df["temperatura_nivel"], categories=["Bajo", "Medio", "Alto"], ordered=True
)
df.boxplot(
    column="costo",
    by="temperatura_nivel",
    ax=axes[1],
    grid=True,
    rot=0,
    boxprops=dict(color="blue", linewidth=1.5),
    medianprops=dict(color="red", linewidth=1.5),
)
axes[1].set_title(
    "Distribución Global de Costos (Boxplot)",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Nivel de Temperatura Inicial", fontsize=11)
axes[1].set_ylabel("Costo Total (Distancia)", fontsize=11)
plt.suptitle("")  # Limpiar el título automático de pandas

plt.tight_layout()
# Se guarda en alta resolución (300 DPI) ideal para el PDF
plt.savefig("doe_factorial_resultados.png", dpi=300, bbox_inches="tight")
plt.show()
print(
    "\n¡Gráfica generada y guardada con éxito como 'doe_factorial_resultados.png'"
    " en tu carpeta!"
)
df_summary = pd.DataFrame(summary_results)
print(df_summary)

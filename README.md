# 🍷 Segmentación de Vinos - RAG — Sistema de análisis y consulta de catálogo de vinos

**Minería de Datos | Universidad Anáhuac Puebla**

Sistema integrado de Machine Learning, almacén de datos (Neon PostgreSQL) y asistente RAG para apoyar decisiones comerciales en una tienda especializada de vinos.

---

## 👥 Equipo

| Integrante | Rol |
|---|---|
| Alejandro Armando Paredes Salcedo | Data Engineer |
| Karla Beatriz Hernandez Castro | Data Scientist |
| Karla Angelica Perez Rodriguez | Solution Architect |
| Melissa Camelo de la Fuente | Data Engineer |

---

## 🎯 Decisor

**Hugo D'Acosta** — Gerente Comercial y enólogo  
**Pregunta de negocio:** ¿Qué vinos priorizar en campañas comerciales según su segmento, probabilidad de rating alto y relación precio-calidad?

---

## 🏗️ Arquitectura del sistema

```
Kaggle CSVs (4 archivos)
        │
        ▼
   EDA + Outliers
        │
        ├─► Clustering K-Means ──────────────┐
        ├─► Random Forest (rating alto) ─────┤──► Neon PostgreSQL
        └─► Minería de texto TF-IDF ─────────┘    (fact_vinos + 3 dims)
                                                          │
                                                          ▼
                                               Documentos RAG (7 docs)
                                                          │
                                               FAISS + sentence-transformers
                                                          │
                                               Ollama (llama3.2 local)
                                                          │
                                               Interfaz ipywidgets
```

---

## 📁 Estructura del repositorio

```
wine-rating-rag/
├── ProyectoFinal_WineRating.ipynb   ← notebook principal
├── requirements.txt
├── .env.example                     ← plantilla de credenciales
├── .env                             ← credenciales reales (NO subir)
├── .gitignore
├── README.md
└── data/                            ← CSVs (no incluidos en el repo)
    ├── Red.csv
    ├── White.csv
    ├── Rose.csv
    └── Sparkling.csv
```

---
### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Descargar los datos

1. Descarga el dataset desde Kaggle: https://www.kaggle.com/datasets/budnyak/wine-rating-and-price
2. Crea la carpeta `data/` y coloca los 4 archivos CSV dentro:
   - `Red.csv`, `White.csv`, `Rose.csv`, `Sparkling.csv`

### Configurar credenciales

Edita el archivo `.env` y agrega tu cadena de conexión de Neon:

```
NEON_URL=postgresql://usuario:password@host.neon.tech/neondb?sslmode=require
```

### Instalar y configurar Ollama

```bash
# Descargar Ollama desde https://ollama.com
# Luego descargar el modelo:
ollama pull llama3.2
```

Ollama se iniciará automáticamente en segundo plano en `localhost:11434`.

### Iniciar JupyterLab

```bash
jupyter lab
```

Abre `ProyectoFinal_Vinos.ipynb` y ejecuta todas las celdas de arriba a abajo.

---

## 🧠 Técnicas de Machine Learning

| Técnica | Objetivo | Resultado |
|---|---|---|
| **K-Means Clustering** | Segmentar vinos | 3 segmentos con silhouette >0.3 |
| **Random Forest** | Predecir rating alto (≥4.2) | ROC-AUC >0.80 |
| **TF-IDF + Regresión Logística** | Minería de texto | Términos predictores de calidad |

---

## 🗄️ Almacén de datos (Neon PostgreSQL)

Esquema estrella con 4 tablas:

| Tabla | Descripción | Filas aprox. |
|---|---|---|
| `fact_vinos` | Vinos con predicciones y segmentos | 13,834 |
| `dim_clusters` | Perfiles de cada segmento | 3 |
| `dim_paises` | Métricas por país | ~40 |
| `dim_metricas` | KPIs del modelo ML | 11 |

---

## 🤖 Asistente RAG

- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones)
- **Índice vectorial:** FAISS (similitud coseno)
- **LLM generativo:** Ollama con llama3.2 local
- **Documentos indexados:** 7 (resultados del análisis ML, no el dataset crudo)

**Preguntas que puede responder:**
1. ¿Qué segmento priorizar en campañas comerciales?
2. ¿Cuáles son los vinos con mayor probabilidad de rating alto?
3. ¿Qué países tienen mejor proporción de vinos bien calificados?
4. ¿Qué términos en el nombre de un vino predicen rating alto?
5. ¿Qué tan confiable es el modelo predictivo?

---

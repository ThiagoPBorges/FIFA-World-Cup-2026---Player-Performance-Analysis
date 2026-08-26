import polars as pl
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "fifa_world_cup_2026_player_performance.csv"

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.POLARS,
  "rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset",
  file_path,
)

pl.Config.set_tbl_rows(-1)

lazy_df = df.lazy()
df_wc = lazy_df.collect()

total_linhas = len(df_wc)
total_colunas = len(df_wc.columns)
colunas_com_nulos = sum(df_wc[col].null_count() > 0 for col in df_wc.columns)

mapa_qualidade = pl.DataFrame({
    "coluna": df_wc.columns,
    "tipo": [str(dtype) for dtype in df_wc.dtypes],
    "pct_nulos": [
        (df_wc[col].null_count() / total_linhas) * 100 
        for col in df_wc.columns
    ],
    "cardinalidade": [
        df_wc[col].n_unique() 
        for col in df_wc.columns
    ]
})

print("--- MAPA DE QUALIDADE ---")
print(f"\nQuantidade de colunas com valores nulos: {colunas_com_nulos}")
print(mapa_qualidade)

print("--- ESTATÍSTICAS ---")
describe_transposto = df_wc.describe().transpose(
    include_header=True, 
    header_name="coluna"
)

with pl.Config(tbl_rows=-1):
    print(describe_transposto)
    
    
print("\n--- T1.1: TAMANHO E MEMÓRIA ---")

memory_mb = df_wc.estimated_size("mb")
print(f"Memória aproximada: {memory_mb:.2f} MB")
print(f"Linhas: {total_linhas:,}")
print(f"Colunas: {total_colunas}")

# Distribuição de tipos
tipos_count = pl.DataFrame({
    "tipo": [str(dtype) for dtype in df_wc.dtypes],
    "coluna": df_wc.columns
}).group_by("tipo").agg(pl.col("coluna").count().alias("quantidade"))

print("\nDistribuição de tipos:")
print(tipos_count)

print("\n💾 Estratégias de redução de memória:")
print("✓ Int64 → Int8/Int16 (se range permite)")
print("✓ Float64 → Float32 (-50% memória)")
print("✓ String → Categorical (baixa cardinalidade)")
print("✓ Drop colunas redundantes")

print("\n--- INVESTIGAÇÃO ---")

print("\n1. Player ratings:")
print(
    df_wc["player_rating"]
    .value_counts()
    .with_columns(
        (pl.col("count") / df_wc.height * 100).round(2).alias("pct")
    )
    .sort(by="count", descending=True)
    .head(10)
)


print("\n2. Registros com passes=0 mas accuracy>0:")
passes_accuracy_issue = df_wc.filter(
    (pl.col("total_passes") == 0) & (pl.col("pass_accuracy") > 0)
)
print(f"Total: {len(passes_accuracy_issue)} registros")
print("\nAmostra:")
print(
    passes_accuracy_issue
    .select("player_name", "total_passes", "pass_accuracy", "minutes_played")
    .head(10)
)

print("\n3. IDs por player_name (qual nome tem múltiplos IDs?):")

ids_por_nome = (
    df_wc.select("player_id", "player_name")
    .unique()  # Remove duplicatas de linha
    .group_by("player_name")
    .agg(pl.col("player_id").count().alias("qtd_ids"))
    .filter(pl.col("qtd_ids") > 1)  # Só nomes com múltiplos IDs
    .sort(by="qtd_ids", descending=True)
)

print(ids_por_nome)

# Ver os detalhes
print("\nDetalhes dos nomes com múltiplos IDs:")
for row in ids_por_nome.iter_rows(named=True):
    nome = row["player_name"]
    ids = df_wc.filter(pl.col("player_name") == nome).select("player_id", "player_name", "team", "club_name", "nationality").unique()
    print(f"\n{nome}:")
    print(ids)
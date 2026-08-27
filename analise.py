import polars as pl
import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Funções Globais ==================

# ------- CARREGAMENTO -------
def carregar_dados():
    
    file_path = "fifa_world_cup_2026_player_performance.csv"

    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.POLARS,
    "rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset",
    file_path,
    )
    
    return df.lazy().collect()

# ------- T1.1 -------
def analise_t11_tamanho_memoria(df):
    
    total_linhas = len(df_wc)
    total_colunas = len(df_wc.columns)
    
    print("\n# ========== T1.1: TAMANHO E MEMÓRIA ==========")

    memory_mb = df.estimated_size("mb")
    print(f"Memória aproximada: {memory_mb:.2f} MB")
    print(f"Linhas: {total_linhas:,}")
    print(f"Colunas: {total_colunas}")

    # Distribuição de tipos
    tipos_count = pl.DataFrame({
        "tipo": [str(dtype) for dtype in df.dtypes],
        "coluna": df.columns
    }).group_by("tipo").agg(pl.col("coluna").count().alias("quantidade"))

    print("\nDistribuição de tipos:")
    print(tipos_count)

    print("\n💾 Estratégias de redução de memória:")
    print("✓ Int64 → Int8/Int16 (se range permite)")
    print("✓ Float64 → Float32 (-50% memória)")
    print("✓ String → Categorical (baixa cardinalidade)")
    print("✓ Drop colunas redundantes")

# ------- T1.2 -------
def analise_t12_qualidade(df):
    
    print("\n# ========== T1.2: MAPA DE QUALIDADE ==========")
    
    total_linhas = len(df_wc)
    colunas_com_nulos = sum(df[col].null_count() > 0 for col in df.columns)
    
    print(f"\nQuantidade de colunas com valores nulos: {colunas_com_nulos}")

    mapa_qualidade = pl.DataFrame({
        "coluna": df.columns,
        "tipo": [str(dtype) for dtype in df.dtypes],
        "pct_nulos": [
            (df[col].null_count() / total_linhas) * 100 
            for col in df.columns
        ],
        "cardinalidade": [
            df[col].n_unique() 
            for col in df.columns
        ]
    })

    print(mapa_qualidade)

    print("--- ESTATÍSTICAS ---")
    describe_transposto = df.describe().transpose(
        include_header=True, 
        header_name="coluna"
    )

    with pl.Config(tbl_rows=-1):
        print(describe_transposto)

    print("\n--- INVESTIGAÇÃO ---")

    print("\n1. Player ratings:")
    print(
        df["player_rating"]
        .value_counts()
        .with_columns(
            (pl.col("count") / df.height * 100).round(2).alias("pct")
        )
        .sort(by="count", descending=True)
        .head(10)
    )


    print("\n2. Registros com passes=0 mas accuracy>0:")
    passes_accuracy_issue = df.filter(
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
        df.select("player_id", "player_name")
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
        ids = df.filter(pl.col("player_name") == nome).select("player_id", "player_name", "team", "club_name", "nationality").unique()
        print(f"\n{nome}:")
        print(ids)

# ------- T1.3 -------
def analise_t13_categorias(df):
    
    print("\n# === T1.3: AS 10 CATEGORIAS ===")

    categorias = {
        "Identificação": ("player_id", "ID único do jogador"),
        "Dados Pessoais": ("height_cm", "Altura do jogador em centímetros"),
        "Posição/Time": ("position", "Posição tática (DF, GK, MF, FW)"),
        "Passes": ("pass_accuracy", "Taxa de passe completado (0-1)"),
        "Defesa": ("tackles", "Desarmes realizados (quantidade)"),
        "Movimentação": ("distance_covered_km", "Distância percorrida (km)"),
        "Velocidade": ("top_speed_kmh", "Velocidade máxima (km/h)"),
        "Ofensiva": ("offensive_contribution", "Contribuição ofensiva (0-100)"),
        "Criatividade/Risco Ofensivo": ("creativity_score", "Capacidade de criar chances perigosas ao time adversário (0-100)"),
        "Valor": ("market_value_eur", "Valor de mercado (EUR)"),
    }

    for i, (categoria, (coluna, descricao)) in enumerate(categorias.items(), 1):
        print(f"{i}. {categoria}: {coluna} → {descricao}")


    print("\n--- GERANDO VISUALIZAÇÕES ---")

    df_pandas = df.to_pandas()

    top_cols = ["creativity_score", "offensive_contribution", "player_rating",
                "goals", "assists", "key_passes", "tackles", "interceptions",
                "distance_covered_km", "top_speed_kmh", "jersey_number",
                "market_value_eur", "pass_accuracy", "consistency_score", "tournament_rating", "possession_impact"]

    corr_matrix = df_pandas[top_cols].corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True)
    plt.title("Matriz de Correlação - FIFA World Cup 2026")
    plt.tight_layout()
    plt.savefig("outputs/heatmap_correlacao.png")
    print("Heatmap salvo: outputs/heatmap_correlacao.png")
    plt.close()

# ================== MAIN ==================
if __name__ == "__main__":
    
    pl.Config.set_tbl_rows(-1)
    
    df_wc = carregar_dados()
    
    # Executa análises
    analise_t11_tamanho_memoria(df_wc)
    analise_t12_qualidade(df_wc)
    analise_t13_categorias(df_wc)
    
    print("\n Análise PARTE 1 (T1.1-T1.3) concluída!")
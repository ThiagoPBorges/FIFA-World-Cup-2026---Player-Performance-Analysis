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

# ==================================== PARTE 01 ====================================

# ------- T1.1 -------
def analise_t11_tamanho_memoria(df):
    
    total_linhas = len(df_wc)
    total_colunas = len(df_wc.columns)
    
    print("\n# ========== T1.1: TAMANHO E MEMÓRIA ==========")

    memory_mb = df.estimated_size("mb")
    print(f"\nMemória aproximada: {memory_mb:.2f} MB")
    print(f"Linhas: {total_linhas:,}")
    print(f"Colunas: {total_colunas}")

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
        "pct_nulos": [(df[col].null_count() / total_linhas) * 100 for col in df.columns],
        "cardinalidade": [df[col].n_unique() for col in df.columns]
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

    print("\n1. Player ratings zerado:")
    
    total_zerados = df.filter(pl.col("player_rating") == 0).height
    player_ratings = (
        df["player_rating"]
        .value_counts()
        .with_columns(
            (pl.col("count") / df.height * 100).round(2).alias("pct")
        )
        .sort(by="count", descending=True)
        .head(10)
    )
    print(f"\nTotal: {total_zerados} registros")
    print(player_ratings)


    print("\n2. Registros com passes=0 mas accuracy > 0:")
    passes_accuracy_issue = df.filter(
        (pl.col("total_passes") == 0) & (pl.col("pass_accuracy") > 0)
    )
    print(f"\nTotal: {len(passes_accuracy_issue)} registros")
    print(
        passes_accuracy_issue
        .select("player_name", "total_passes", "pass_accuracy", "minutes_played")
        .head(10)
    )

    print("\n3. Players com múltiplos IDs:")
    
    # 1. Resumo agrupado
    ids_por_nome = (
        df.group_by("player_name")
        .agg(
            pl.col("player_id").n_unique().alias("qtd_ids"),
            pl.col("club_name").n_unique().alias("qtd_teams"),
            pl.col("club_name").unique().alias("clubs"),
        )
        .filter(pl.col("qtd_ids") > 1)
        .sort(by="qtd_ids", descending=True)
    )
    
    print(f"Total de nomes duplicados: {ids_por_nome.height}")
    print(ids_por_nome)

# ------- T1.3 -------
def analise_t13_categorias(df):
    
    print("\n# === T1.3: AS 10 CATEGORIAS ===")

    categorias = {
        "Gols Esperados": ("expected_goals_xg", "Qualidade das chances criadas (escala 0-5, valores decimais)"),
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

# ------- T1.4 -------
def analise_t14_coerencia(df):
    
    print("\n# ========== T1.4: REGRAS DE COERÊNCIA ==========")
    print("\nValidando 6 categorias de incoerências\n")
    
    total_violacoes = 0

    # ===== CATEGORIA 1: MATEMÁTICA =====
    print("CATEGORIA 1: INCOERÊNCIAS MATEMÁTICAS")
    
    print("\n  Jogadores com multiplas partidas no mesmo dia")
    print("\n  Regra 1.0: player > 1 match_id por match_date")
    v0 = (
        df.group_by("player_id", "player_name", "match_date")
        .agg(pl.col("match_id").n_unique().alias("qtd_matches_no_dia"))
        .filter(pl.col("qtd_matches_no_dia") > 1)
    )
    print(f"Violações: {len(v0)}")
    if len(v0) > 0:
        print(v0.select("player_name", "match_date", "qtd_matches_no_dia").sort("qtd_matches_no_dia", descending=True).head())
    total_violacoes += len(v0)    

    print("\n  Passes bem-sucedidos não podem ser maiores que passes totais")
    print("\n  Regra 1.1: successful_passes ≤ total_passes")
    v2 = df.filter(pl.col("successful_passes") > pl.col("total_passes"))
    print(f"Violações: {len(v2)}")
    total_violacoes += len(v2)
    
    print("\n  Regra 1.3: Muitos jogos em 'Final' (deve ser 1-2 no máximo)")
    # Final deveria ter poucos jogos (só grandes decisões)
    final_games = df.filter(pl.col("tournament_stage") == "Final")
    qtd_final = len(final_games)
    print(f"Quantidade de jogos em 'Final': {qtd_final}")
    
    if qtd_final > 2:
        print(
            final_games.select("player_name", "team", "match_date", "tournament_stage")
            .unique()
            .sort("match_date", descending=True)
            .head(5)
        )
    total_violacoes += max(0, qtd_final - 2)  # Penaliza o excesso

    print("\n  Regra 1.4: Último jogo não é a Final (inconsistência temporal)")
    ultimo_jogo = (
        df.sort("match_date", descending=True)
        .select("match_date", "tournament_stage")
        .unique()
        .head(1)
    )
    
    if len(ultimo_jogo) > 0:
        ultimo_date = ultimo_jogo["match_date"][0]
        ultimo_stage = ultimo_jogo["tournament_stage"][0]
        
        print(f"Último jogo (data): {ultimo_date} → Stage: {ultimo_stage}")
        
        if ultimo_stage != "Final":
            print(f"VIOLAÇÃO: Último jogo deveria ser 'Final', não '{ultimo_stage}'")
            total_violacoes += 1
        else:
            print(f"Correto: Último jogo é a Final")
    else:
        print("Sem dados!")

    # ===== CATEGORIA 2: FÍSICA =====
    print("\n CATEGORIA 2: INCOERÊNCIAS FÍSICAS")
    
    print("\n  Regra 2.1: sprint_distance_km ≤ distance_covered_km")
    v4 = df.filter(pl.col("sprint_distance_km") > pl.col("distance_covered_km"))
    print(f"Violações: {len(v4)}")
    total_violacoes += len(v4)

    # ===== CATEGORIA 3: ESTATÍSTICA =====
    print("\n CATEGORIA 3: INCOERÊNCIAS ESTATÍSTICAS")
    
    print("\n  Regra 3.1: Expected Goals vs Actual Goals (desvio > 2)")
    v5 = df.filter(((pl.col("goals") - pl.col("expected_goals_xg")).abs() > 2))
    print(f"   Anomalias: {len(v5)}")
    if len(v5) > 0:
        print(v5.select("player_name", "goals", "expected_goals_xg").head())
    total_violacoes += len(v5)

    # ===== CATEGORIA 4: POSIÇÃO =====
    print("\n CATEGORIA 4: INCOERÊNCIAS POR POSIÇÃO")
    
    print("\n  Regra 4.1: Goleiro com criatividade alta (> 50)")
    v6 = df.filter((pl.col("position") == "GK") & (pl.col("creativity_score") > 50))
    print(f"   Anomalias: {len(v6)}")
    if len(v6) > 0:
        print(v6.select("player_name", "position", "creativity_score").head())
    total_violacoes += len(v6)

    # ===== CATEGORIA 5: EXPERIÊNCIA =====
    print("\n CATEGORIA 5: INCOERÊNCIAS DE EXPERIÊNCIA")
    
    print("\n  Regra 5.1: Jogador jovem (≤17) com muitos minutos (>500)")
    v7 = df.filter((pl.col("age") <= 17) & (pl.col("total_minutes_tournament") > 500))
    print(f"   Anomalias: {len(v7)}")
    if len(v7) > 0:
        print(v7.select("player_name", "age", "total_minutes_tournament").head())
    total_violacoes += len(v7)

    # ===== CATEGORIA 6: CONSISTÊNCIA =====
    print("\n✓ CATEGORIA 6: INCOERÊNCIAS DE CONSISTÊNCIA")
    
    print("\n  Regra 6.1: Minutos por jogo ≤ 90")
    v8 = (
        df.group_by("player_id", "player_name")
        .agg((pl.col("minutes_played").sum() / pl.col("match_id").count()).alias("min_por_jogo"))
        .filter(pl.col("min_por_jogo") > 90)
    )
    print(f"Violações: {len(v8)}")
    total_violacoes += len(v8)

    # ===== RESUMO POR POSIÇÃO =====
    print("\n  Rating médio por posição")
    media_rating = df.group_by("position").agg(
        pl.col("player_rating").mean().round(2).alias("rating_médio")
    ).sort("rating_médio", descending=True)
    print(media_rating)

    # ===== RESUMO FINAL =====
    print("\n" + "="*60)
    print(f" TOTAL DE INCOERÊNCIAS DETECTADAS: {total_violacoes}")
    print("="*60)
    if total_violacoes == 0:
        print("Dados passaram em TODAS as validações")
    else:
        print(f"{total_violacoes} problemas detectados")

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

# ------- T2.1 -------
def analise_t21_metricas_90_min(df):
    
    print("\n# ========== T2.1: MÉTRICAS POR 90 MINUTO ==========")
    
    df_90min = df.select([
        "player_id",
        "player_name",
        "position",
        "team",
        "minutes_played",
        "goals",
        "assists",
        "key_passes",
        "shots",
        "tackles",
        "distance_covered_km"
    ]).with_columns([
        # Evita divisão por zero
        (pl.when(pl.col("minutes_played") > 0)
         .then((pl.col("goals") * 90.0) / pl.col("minutes_played"))
         .otherwise(0)
         .alias("goals_per90")),
        
        (pl.when(pl.col("minutes_played") > 0)
         .then((pl.col("assists") * 90.0) / pl.col("minutes_played"))
         .otherwise(0)
         .alias("assists_per90")),
        
        (pl.when(pl.col("minutes_played") > 0)
         .then((pl.col("key_passes") * 90.0) / pl.col("minutes_played"))
         .otherwise(0)
         .alias("key_passes_per90")),
        
        (pl.when(pl.col("minutes_played") > 0)
         .then((pl.col("shots") * 90.0) / pl.col("minutes_played"))
         .otherwise(0)
         .alias("shots_per90")),
        
        (pl.when(pl.col("minutes_played") > 0)
         .then((pl.col("tackles") * 90.0) / pl.col("minutes_played"))
         .otherwise(0)
         .alias("tackles_per90")),
        
        (pl.when(pl.col("minutes_played") > 0)
         .then((pl.col("distance_covered_km") * 90.0) / pl.col("minutes_played"))
         .otherwise(0)
         .alias("distance_per90_km"))
    ]).unique()
    
    print("Primeiros 10 jogadores com métricas por 90min:")
    print(
        df_90min.select([
            "player_name", "position", "goals_per90", "assists_per90", 
            "key_passes_per90", "distance_per90_km"
        ])
        .sort("goals_per90", descending=True)
        .head(10)
    )

    return df_90min

# ==================================== PARTE 02 ====================================

# ------- T2.2 -------
def analise_t22_artilheiro_90_min(df_90min):
    
    print("\n# ========== T2.2: TOP 10 ARTILHEIROS ==========\n")
    
    top10_shooters = (
        df_90min
        .filter(pl.col("minutes_played") >= 90)
        .sort("goals_per90", descending=True)
        .head(10)
        .select([
            "player_name", "team", "position", "minutes_played", 
            "goals", "goals_per90", "shots", "shots_per90"
        ])
    )
    
    print("Top 10 (mínimo 90 minutos):")
    print(top10_shooters)

# ------- T2.3 -------
def analise_t23_agregar_por_selecao(df):
    print("\n# ========== T2.3: AGREGADO POR SELEÇÃO ==========\n")
    
    agregado_pais = (
        df.group_by("team")
        .agg([
            pl.col("player_id").n_unique().alias("num_jogadores_unicos"),
            pl.col("goals").sum().alias("gols_total"),
            pl.col("assists").sum().alias("assists_total"),
            pl.col("player_rating").mean().round(2).alias("rating_médio"),
            pl.col("distance_covered_km").sum().round(2).alias("distância_total_km"),
            pl.col("creativity_score").mean().round(2).alias("criatividade_média"),
            pl.col("offensive_contribution").mean().round(2).alias("ofensiva_média")
        ])
        .sort("gols_total", descending=True)
    )
    
    print("Top 10 seleções:")
    print(agregado_pais.head(10))
    
    return agregado_pais  # ← RETORNA

# ------- T2.4 -------
def analise_t24_goals_xg(df):
    print("\n# ========== T2.4: GOALS vs xG ==========\n")
    
    df_performance = (
        df.select([
            "player_id", "player_name", "team", "position", 
            "goals", "expected_goals_xg", "shots", "shots_on_target"
        ])
        .unique()
        .with_columns([
            (pl.col("goals") - pl.col("expected_goals_xg")).alias("over_performance")
        ])
    )
    
    print("\nover_performance = goals − xG.\n")
    
    print("TOP FINALIZADORES (over_performance > 0.5):")
    print(
        df_performance
        .filter((pl.col("over_performance") > 0.5) & (pl.col("expected_goals_xg") > 1))
        .sort("over_performance", descending=True)
        .head(10)
        .select([
            "player_name", "team", "goals", "expected_goals_xg", "over_performance"
        ])
    )

    # ===== NOVO: QUEM DESPERDIÇA =====
    print("\n\nTOP DESPERDIÇADORES (over_performance < -0.5):")
    desperdicadores = (
        df_performance
        .filter((pl.col("over_performance") < -0.5) & (pl.col("expected_goals_xg") > 1))
        .sort("over_performance", descending=False)
        .head(10)
        .select([
            "player_name", "team", "goals", "expected_goals_xg", "over_performance"
        ])
    )
    print(desperdicadores)

    return {
        "performance": df_performance,
        "desperdicadores": desperdicadores
    }

# ------- T2.5 -------
def dados_tratados_parquet(df):
    
    print("\n# ========== LIMPEZA E PARQUET ==========\n")
    
    
    print("1️Removendo colunas agregadas.")
    df_gold = df.drop([
        "total_minutes_tournament",
         "total_goals_tournament",
         "total_assists_tournament"
    ])
    
    # SALVAR base BRUTA (backup)
    print("\n3️Salvando base BRUTA (backup)")
    df.write_parquet("data/raw.parquet")
    print("Salvo: data/01_bruto_raw.parquet")
    
    # SALVAR base TRATADA (sem colunas ruins)
    print("\n4️Salvando base TRATADA (colunas agregadas removidas)")
    df_gold.write_parquet("data/gold.parquet")
    tamanho_gold = df_gold.estimated_size("mb")
    print(f"Salvo: data/02_tratado.parquet ({tamanho_gold:.1f} MB)")
    
    # 5. ESTATÍSTICAS
    print("\n5️Resumo da limpeza:")
    print(f"Registros: {len(df):,} (mantidos)")
    print(f"Colunas removidas: 3")
    print(f"Colunas restantes: {len(df_gold)}")
    print(f"Linhas perdidas: 0 (nenhuma removida, só colunas)")
    
    print("\n⚠️  DADOS COM MÚLTIPLOS JOGOS/DIA:")
    print("- 13.546 registros com >1 jogo no mesmo dia")
    print("- Impacto: Métricas por 90min podem estar enviesadas")
    
    return df_gold

# ==================================== PARTE 03 ====================================

# ------- T3.1 -------
def analise_t31_scatter_xg_goals(df):
    print("\n# ========== T3.1: SCATTER xG × GOALS ==========\n")
    
    # Preparar dados
    df_scatter = (
        df.select(["player_name", "team", "position", "goals", "expected_goals_xg"])
        .unique()
        .filter(pl.col("expected_goals_xg") > 0)  # Só quem tem xG
    ).to_pandas()
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(df_scatter["expected_goals_xg"], df_scatter["goals"], 
               alpha=0.6, s=100, c="steelblue", edgecolors="black", linewidth=0.5)
    
    # Linha y=x (referência: goals = xG esperado)
    max_val = max(df_scatter["expected_goals_xg"].max(), df_scatter["goals"].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="y=x (esperado)")
    
    # Rotular OUTLIERS (maiores desvios)
    df_scatter["desvio"] = abs(df_scatter["goals"] - df_scatter["expected_goals_xg"])
    outliers = df_scatter.nlargest(10, "desvio")
    
    for idx, row in outliers.iterrows():
        ax.annotate(row["player_name"], 
                   (row["expected_goals_xg"], row["goals"]),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Expected Goals (xG)", fontsize=12)
    ax.set_ylabel("Goals Reais", fontsize=12)
    ax.set_title("Over-Performance: Goals vs Expected Goals\n(Acima da linha = bom finalizador, Abaixo = desperdiçador)", 
                fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/t31_scatter_xg_goals.png", dpi=300)
    print("Salvo: outputs/t31_scatter_xg_goals.png")
    plt.close()

# ------- T3.2 -------
def analise_t32_rating_por_posicao(df):
    print("\n# ========== T3.2: RATING POR POSIÇÃO ==========\n")
     
    df_rating = (
        df.select(["position", "player_rating"])
        .filter(pl.col("player_rating") > 0)  # Remove 0s
        .unique()
    ).to_pandas()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Violino (melhor que boxplot para distribuição)
    sns.violinplot(data=df_rating, x="position", y="player_rating", 
                   palette="Set2", ax=ax)
    
    # Adicionar pontos (jitter)
    sns.stripplot(data=df_rating, x="position", y="player_rating", 
                  color="black", alpha=0.3, size=3, ax=ax)
    
    ax.set_xlabel("Posição", fontsize=12)
    ax.set_ylabel("Player Rating (0-10)", fontsize=12)
    ax.set_title("Distribuição de Rating por Posição\nQual posição tem maior dispersão?", 
                fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("outputs/t32_rating_posicao.png", dpi=300)
    print("Salvo: outputs/t32_rating_posicao.png")
    plt.close()
    
    # Estatísticas
    print("\nEstatísticas por posição:")
    stats = df_rating.groupby("position")["player_rating"].agg([
        ("média", "mean"), 
        ("mediana", "median"), 
        ("desvio", "std"),
        ("min", "min"),
        ("max", "max")
    ]).round(2)
    print(stats)

# ------- T3.3 -------
def analise_t33_pizza_vs_barras(df):
    print("\n# ========== T3.3: PIZZA vs BARRAS ==========\n")
     
    # Agregado por país (top 24)
    df_paises = (
        df.group_by("team")
        .agg(pl.col("goals").sum().alias("gols"))
        .sort("gols", descending=False)
        .head(24)
    ).to_pandas()
    
    # Criar 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.Set3(range(len(df_paises)))
    
    # ===== PIZZA 2D (esquerda) =====
    wedges, texts, autotexts = ax1.pie(
        df_paises["gols"],
        labels=df_paises["team"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )
    ax1.set_title("Pizza 2D\n(Difícil comparar valores)", fontsize=12, fontweight="bold")
    
    # Melhorar legibilidade
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)
    
    # ===== BARRAS (direita) =====
    bars = ax2.barh(df_paises["team"], df_paises["gols"], color=colors, edgecolor="black")
    ax2.set_xlabel("Gols", fontsize=11, fontweight="bold")
    ax2.set_title("Bar Chart\n(Fácil comparar valores)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    
    # Adicionar valores nas barras
    for i, (team, gols) in enumerate(zip(df_paises["team"], df_paises["gols"])):
        ax2.text(gols + 0.5, i, f"{gols}", va="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("outputs/t33_pizza_vs_barras.png", dpi=300, bbox_inches="tight")
    print("✅ Salvo: outputs/t33_pizza_vs_barras.png")
    
    print("\nANÁLISE:")
    print("Pizza: Mostra percentual, mas difícil ler valores absolutos")
    print("Barras: Fácil comparar valores e ver ranking")
    
    plt.close()

# ------- T3.4 -------
def analise_t34_bonus_frontend(df):
    print("\n# ========== T3.4: BÔNUS FRONT-END ==========\n")
    
    # Preparar dados: Top 20 artilheiros
    df_top = (
        df.select(["player_name", "team", "position", "goals", "expected_goals_xg"])
        .unique()
        .filter(pl.col("goals") > 0)
        .sort("goals", descending=True)
        .head(20)
    ).to_pandas()
    
    # Converter para JSON
    data_json = df_top.to_json(orient="records")
    
    # HTML interativo
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Top 20 Artilheiros - Copa 2026</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            h1 {{ text-align: center; color: white; margin-bottom: 30px; }}
            .search-box {{ text-align: center; margin-bottom: 20px; }}
            input {{ padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; width: 300px; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 8px 16px rgba(0,0,0,0.2); }}
            th {{ background: #667eea; color: white; padding: 15px; text-align: left; font-weight: 600; }}
            td {{ padding: 12px 15px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background: #f5f5f5; }}
            .rank {{ font-weight: bold; color: #667eea; }}
            .over-perf {{ color: green; font-weight: bold; }}
            .under-perf {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚽ Top 20 Artilheiros - Copa do Mundo 2026</h1>
            
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Procurar por jogador, time ou posição..." onkeyup="filterTable()">
            </div>
            
            <table id="dataTable">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Jogador</th>
                        <th>Time</th>
                        <th>Posição</th>
                        <th>Gols</th>
                        <th>xG</th>
                        <th>Over-Performance</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                </tbody>
            </table>
        </div>
        
        <script>
            const data = {data_json};
            
            function renderTable(filteredData) {{
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                filteredData.forEach((player, index) => {{
                    const overPerf = (player.goals - player.expected_goals_xg).toFixed(2);
                    const perfClass = overPerf > 0 ? 'over-perf' : 'under-perf';
                    
                    const row = `
                        <tr>
                            <td class="rank">#{{index + 1}}</td>
                            <td><strong>{{player.player_name}}</strong></td>
                            <td>{{player.team}}</td>
                            <td>{{player.position}}</td>
                            <td><strong>{{player.goals}}</strong></td>
                            <td>{{player.expected_goals_xg.toFixed(2)}}</td>
                            <td class="{{perfClass}}">{{overPerf}}</td>
                        </tr>
                    `;
                    tbody.innerHTML += row;
                }});
            }}
            
            function filterTable() {{
                const input = document.getElementById('searchInput').value.toLowerCase();
                const filtered = data.filter(player => 
                    player.player_name.toLowerCase().includes(input) ||
                    player.team.toLowerCase().includes(input) ||
                    player.position.toLowerCase().includes(input)
                );
                renderTable(filtered);
            }}
            
            // Renderizar inicial
            renderTable(data);
        </script>
    </body>
    </html>
    """
    
    # Salvar HTML
    with open("outputs/t34_ranking_interativo.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Salvo: outputs/t34_ranking_interativo.html")
    print("   Abra no navegador para usar a busca interativa!")


# ==================================== PARTE 04 ====================================

def analise_t41_jogadores_subvalorizados(df):
    print("\n# ========== T4.1: JOGADORES SUBVALORIZADOS ==========\n")
    
    print("CRITÉRIO:")
    print("Performance ACIMA da média + Rating ABAIXO da média\n")
    
    # 1. Calcular MÉDIA POR POSIÇÃO
    media_posicao = (
        df.select(["position", "goals", "assists", "player_rating"])
        .unique()
        .group_by("position")
        .agg([
            pl.col("goals").mean().round(2).alias("media_goals"),
            pl.col("assists").mean().round(2).alias("media_assists"),
            pl.col("player_rating").mean().round(2).alias("media_rating")
        ])
    )
    
    print("MÉDIAS POR POSIÇÃO:")
    print(media_posicao)
    print()
    
    # 2. Comparar CADA JOGADOR com a média da sua posição
    df_com_media = (
        df.select(["player_name", "team", "position", "goals", "assists", 
                   "player_rating", "market_value_eur"])
        .unique()
        .join(media_posicao, on="position")
    )
    
    # 3. SUBVALORIZADOS = Performance boa + Rating baixo
    subvalorizado = (
        df_com_media
        .filter(
            (pl.col("goals") > pl.col("media_goals")) &      # Gols > média
            (pl.col("assists") >= pl.col("media_assists")) & # Assists ≥ média
            (pl.col("player_rating") < pl.col("media_rating"))  # Rating < média
        )
        .with_columns([
            (pl.col("goals") - pl.col("media_goals")).alias("gols_acima_media"),
            (pl.col("media_rating") - pl.col("player_rating")).alias("rating_abaixo_media")
        ])
        .select([
            "player_name", "team", "position", 
            "goals", "media_goals", "gols_acima_media",
            "player_rating", "media_rating", "rating_abaixo_media",
            "market_value_eur"
        ])
        .sort("market_value_eur")  # Barato = melhor subvalorizado
        .head(10)
    )
    
    print("✅ JOGADORES SUBVALORIZADOS (10 mais baratos):")
    print(subvalorizado)

def analise_t43_pensamento_critico_sprints(df):
    
    print("\n# ========== T4.3: PENSAMENTO CRÍTICO ==========\n")
    
    print("CENÁRIO:")
    print("Colega encontrou: Correlação 0.8 entre Sprints e Rating")
    print("Quer publicar: 'Correr mais melhora a nota'\n")
    
    print("="*70)
    print("OBJEÇÃO (Por que NÃO publicar isso):")
    print("="*70 + "\n")
    
    print("CORRELAÇÃO ≠ CAUSALIDADE")
    print("Correlação 0.8 é ALTA, mas não prova causa-efeito!\n")
    
    print("TRÊS EXPLICAÇÕES POSSÍVEIS:\n")
    
    print("A) Correr mais → Rating melhor (O que ele pensa)")
    print("Problema: Por que correr melhoraria a nota?")
    print("Goleiros correm POUCO mas têm bom rating!\n")
    
    print("B) Rating melhor → Correr mais (Causa reversa)")
    print("Jogadores bons naturalmente correm mais")
    print("Rating não causa isso, é consequência!\n")
    
    print("C) TERCEIRA VARIÁVEL causa AMBOS (PROVÁVEL)")
    print("Posição do jogador!")
    print("Atacantes: correm MUITO + rating ALTO")
    print("Meias: correm MÉDIO + rating MÉDIO")
    print("Defensores: correm POUCO + rating VARIA")
    print("Goleiros: correm MUITO POUCO + rating ESPECIALIZADO\n")
    
    print("="*70)
    print("COMO VALIDAR CORRETAMENTE:")
    print("="*70 + "\n")
    
    # Teste controlado: Correlação POR POSIÇÃO
    print("Calcular correlação separadamente para cada posição:\n")
    
    df_teste = (
        df.select(["position", "sprint_distance_km", "minutes_played", "player_rating"])
        .unique()
        .filter(pl.col("minutes_played") > 0)
        .with_columns([
            (pl.col("sprint_distance_km") * 90 / pl.col("minutes_played"))
            .alias("sprints_per_90")
        ])
    )
    
    # Correlação por posição
    for posicao in df_teste.select("position").unique().to_series().to_list():
        df_pos = (
            df_teste.filter(pl.col("position") == posicao)
            .to_pandas()
        )
        
        if len(df_pos) > 2:  # Precisa mínimo 3 pontos
            corr = df_pos["sprints_per_90"].corr(df_pos["player_rating"])
            print(f"   {posicao}: Correlação = {corr:.3f}")
    
    print("\nSE a correlação DESAPARECER = Prova que era a POSIÇÃO!")
    print("SE a correlação MANTIVER = Aí sim há relação real\n")
    
    print("="*70)
    print("CONCLUSÃO PARA PUBLICAR:")
    print("="*70 + "\n")
    
    print("NÃO escreva: 'Correr mais melhora a nota'")
    print("Motivo: Não pode provar causalidade com dados observacionais\n")
    
    print("ESCREVA: 'Correlação entre Sprints e Rating (0.8) pode ser explicada por diferenças de posição.")
    print("É recomendado uma análise controlada por posição.'")
    
    return {
        "correlacao_geral": 0.8,
        "msg": "Correlação ≠ Causalidade. Sempre controle variáveis confundidoras!"
    }

# ================== MAIN ==================
if __name__ == "__main__":
    
    pl.Config.set_tbl_rows(-1)
    
    df_wc = carregar_dados()
    
    # PARTE 1
    #analise_t11_tamanho_memoria(df_wc)
    #analise_t12_qualidade(df_wc)
    #analise_t13_categorias(df_wc)
    #analise_t14_coerencia(df_wc)

    # PARTE 2
    #df_90min = analise_t21_metricas_90_min(df_wc)
    #analise_t22_artilheiro_90_min(df_90min)
    #agregado_pais = analise_t23_agregar_por_selecao(df_wc)
    #df_performance = analise_t24_goals_xg(df_wc)
    #dados_tratados_parquet(df_wc)
    
    # PARTE 3
    #analise_t31_scatter_xg_goals(df_wc)
    #analise_t32_rating_por_posicao(df_wc)
    #analise_t33_pizza_vs_barras(df_wc)
    #analise_t34_bonus_frontend(df_wc)
    
    # PARTE 4
    #analise_t41_jogadores_subvalorizados(df_wc)
    analise_t43_pensamento_critico_sprints(df_wc)
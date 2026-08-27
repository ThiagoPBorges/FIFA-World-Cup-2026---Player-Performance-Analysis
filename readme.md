## Parte 0 - Baixar o dataset

#### 1. Qual caminho usou e por quê? Como garante que o arquivo veio íntegro?

* **Caminho Utilizado:** **Caminho C (`kagglehub`)**
* **Por quê:** Garante a automação e reprodutibilidade do pipeline sem dependências de downloads manuais. O kagglehub faz a gestão de cache local, além de permitir o uso nativo do Polars, otimizando o uso de memória RAM.
* **Garantia de Integridade:** O kagglehub valida o arquivo comparando um código único (checksum) do arquivo baixado com o do servidor Kaggle. Se forem idênticos, o arquivo chegou intacto; se differirem, há corrupção ou alteração nos dados, garantindo confiabilidade para a análise.

#### 2. Onde as credenciais (`kaggle.json`) não devem ser guardadas — e por quê?

* **Onde NÃO guardar:** Dentro da pasta do projeto versionada pelo Git, nem commitadas em repositórios (públicos ou privados).
* **Por quê:** O arquivo kaggle.json contém a chave privada da API da conta. Se for exposto, terceiros podem utilizar a  conta indevidamente. As credenciais devem ser mantidas no diretório raiz do usuário ou passadas via variáveis de ambiente.

#### 3. O que a licença MIT permite e exige ao publicar um resultado?

* **Permissões:** É uma licença de permissão, que concede o direito de usar, copiar, modificar, publicar, distribuir cópias do código/dados sem restrições.
* **Exigências:** Exige apenas que o aviso de crédito original e a declaração de permissão da licença MIT sejam mantidos e incluídos em todas as cópias ou partes do projeto publicado.

#### 4. O que deve e o que não deve ser versionado no Git neste projeto?

* **O que DEVE ser versionado:**
  * Códigos-fonte (`Análises` e `scripts de ETL`).
  * Arquivos de configuração de dependências (`requirements.txt`).
  * Arquivo `.gitignore`.
  * Documentação (`README.md`).
* **O que NÃO deve ser versionado:**
  * O arquivo de dados baixado (`fifa_world_cup_2026_player_performance.csv`), pois o pipeline deve ser capaz de baixá-lo via script e arquivos grandes não devem ir para o Git.
  * Credenciais e tokens (`kaggle.json`, `.env`).
  * Pasta do ambiente virtual (`venv`).
  * Arquivos temporários e caches (`__pycache__`).

## Parte 1 - Primeiro Contato / Sanidade

### T1.1 - Tamanho e Memória

**Achados:**
- Memória: 31.55 MB (gerenciável)
- Linhas: 54.600 | Colunas: 75
- Distribuição: 42 Int64, 28 Float64, 5 String
- **Potencial redução:** Float64→Float32 = -50% (13.7 MB)

---

### T1.2 - Mapa de Qualidade: 3 Colunas Suspeitas

**Coluna 1: `player_rating`**

- **Suspeita:** 42.2% dos 54.600 registros têm valor 0.0 (23.042 casos)
- **Raciocínio:** Um rating de 0 é Improvável para um jogador que atuou
- **Validação:** Verificação à correlação com minutes_played = 0 (possível missing)

**Coluna 2: `pass_accuracy`**

- **Suspeita:** 23.096 registros têm pass_accuracy > 0 mas total_passes = 0
- **Raciocínio:** Violação lógica — acurácia sem passes realizados 
- **Descoberta:** Todos esses casos têm minutes_played = 0 (subs que não entraram)
- **Validação:** Padrão esperado; dados de substitutos com estatísticas default

**Coluna 3: `player_id` duplicado**

- **Suspeita:** 3 nomes com múltiplos player_ids
  - Pierre-Emile Christensen: P00262 e P00274
  - Hee-chan Hwang: P00834 e P00837
  - Kalidou Mendy: P01016 e P01019
- **Raciocínio:** Pode ser duplicação ou erro de ID assignment
- **Validação:** Mesmo time, mesma nacionalidade — possível duplicação do dataset

### T1.3 - As 10 Categorias

1. **Identificação** → `player_id` — ID único do jogador
2. **Dados Pessoais** → `height_cm` — Altura em centímetros
3. **Posição/Time** → `position` — Posição tática (DF, GK, MF, FW)
4. **Passes** → `pass_accuracy` — Taxa de passe (0-1)
5. **Defesa** → `tackles` — Desarmes realizados
6. **Movimentação** → `distance_covered_km` — Distância em km
7. **Velocidade** → `top_speed_kmh` — Velocidade máxima
8. **Ofensiva** → `offensive_contribution` — Contribuição no ataque (0-100)
9. **Criatividade** → `creativity_score` — Capacidade de criar chances (0-100)
   - Correlação forte (0.727) com `offensive_contribution`
   - Jersey_number (0.569): Atacantes > Defensores
10. **Valor de Mercado** → `market_value_eur` — Valor em euros

---

### Visualizações Geradas

- `outputs/heatmap_correlacao.png` — Matriz de correlações entre 16 métricas-chave

## Referências e Recursos de Estudo

- [Checksum e Integridade de Arquivos](https://www.youtube.com/watch?v=Xlevzwy5_no) — Vídeo que ajudou a entender a validação de integridade do kagglehub.

## Atribuição

**Dataset:** fifa_world_cup_2026_player_performance.csv
**Autor Original:** rauffauzanrambe
**Fonte:** Kaggle
**Licença:** MIT

Conforme os termos da licença MIT, este projeto mantém a atribuição ao criador original do dataset.
Link - [www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset)

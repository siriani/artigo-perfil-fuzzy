# fuzzyprofile — Fuzzy C-Means para perfis de personalidade Big Five (IPIP-50)

Implementação de referência e scripts de experimento para agrupamento difuso de perfis
psicométricos. Este repositório contém **apenas o código**, para que outros
pesquisadores possam reproduzir os experimentos. O manuscrito é mantido à parte.

> **Reference implementation for fuzzy clustering of Big Five (IPIP-50) personality
> profiles.** From-scratch Fuzzy C-Means, cluster-validity indices, a boundary
> distortion rate `ρ(c, m)`, Likert-uncertainty encodings (TFN / IFS), and benchmarks
> against k-means, Gustafson–Kessel, PCM and a Gaussian mixture. Code only.

## Conteúdo

```
src/fuzzyprofile/
  engine.py         FuzzyCMeansEngine — FCM de Bezdek (Euclidiana / Mahalanobis)
  validity.py       FPC, MPC, PE, Xie–Beni, Fukuyama–Sugeno, Kwon (vetorizados)
  distances.py      distâncias ao quadrado + regularização de covariância
  preprocess.py     escores fatoriais OCEAN, PCA esférico, diagnóstico de condicionamento
  likert_fuzzify.py codificação TFN / IFS de respostas Likert
  data.py           carregador do IPIP-50 real + gerador sintético
  gridsearch.py     busca em grade (c, m) + seleção
  benchmarks.py     hard c-means, Gustafson–Kessel, PCM, GMM + compare()
experiments/
  _common.py        carga de dados e configuração compartilhada
  run_all.py        pipeline: grade (c,m), 3 projeções, comparativo, GMM, figuras
  run_gridsearch.py / run_benchmark.py   etapas isoladas
  run_phase6.py     ablação Likert, estabilidade sob reamostragem, protótipos
  figures.py        geração de figuras
scripts/get_data.sh  baixa o dataset IPIP-50 (domínio público, sem login)
```

Saídas (CSV, JSON, PNG) são escritas em `results/` (não versionado).

## Instalação

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # numpy scipy pandas scikit-learn matplotlib
```

## Dados

`scripts/get_data.sh` baixa `IPIP-FFM-data-8Nov2018` do Open-Source Psychometrics
Project (mesmo arquivo do Kaggle `tunguz/big-five-personality-test`), domínio público.
Detalhes de proveniência e chave de itens reversos em [`data/README.md`](data/README.md).
Sem o download, todos os scripts recaem em dados sintéticos (com aviso).

## Reproduzir os experimentos

```bash
bash scripts/get_data.sh
python -m src.fuzzyprofile.engine        # autoteste do motor FCM
python experiments/run_all.py            # ~6 min; grade + comparativo + figuras + results/summary.json
python experiments/run_phase6.py         # ~18 min; ablação + reamostragem + protótipos
```

Configuração fixa (em `experiments/_common.py`):

| Parâmetro | Valor |
|---|---|
| Subamostra | `n = 25 000` sorteados de todos os 603 322 registros completos com `IPC == 1` |
| Semente | 42 (execução principal); 1000..1019 (bootstraps da Fase 6) |
| Grade | `c ∈ {2,…,10}`, `m ∈ {1,1; 1,2; …; 3,0}` (180 células) |
| Inicialização | k-means++ (`D²`), 2 reinícios na grade, 5 no ponto de operação |
| Parada | `‖ΔU‖∞ < 10⁻⁴` **ou** `|ΔJ_m|/J_m < 10⁻⁷` **ou** `k = 300` |
| Ponto de operação | `c = 6`, `m = 2,0` |
| Espaços de atributos | OCEAN-5 (Euclidiana e Mahalanobis), PCA esférico (95 % var.), 50 itens (controle) |

Resultado central esperado: no fuzzifier padrão `m = 2`, a taxa de distorção de
fronteira `ρ(c, m) = n⁻¹·|{j : maxᵢ uᵢⱼ < 0,5}| ≈ 1` para `c ≥ 3` nas três
representações; um controle sintético com 5 agrupamentos genuínos mantém `ρ = 0` no
mesmo cenário.

## Licença

**MIT** (ver [`LICENSE`](LICENSE)).

## Autores

Nívea Regina Marques Aguiar, Renato Dias Baptista, Allan Lincoln Rodrigues Siriani,
Luís Roberto Almeida Gabriel Filho — Programa de Pós-Graduação em Agronegócio e
Desenvolvimento, FCE/UNESP, Tupã, SP.

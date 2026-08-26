# Dataset — IPIP-50 (Big Five / OCEAN)

## Fonte primária

**Open-Source Psychometrics Project — IPIP-FFM-data-8Nov2018**
<http://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip> (sem login).

O mesmo arquivo é distribuído no Kaggle como
[`tunguz/big-five-personality-test`](https://www.kaggle.com/datasets/tunguz/big-five-personality-test)
e [`sarmentor/big-5-personality-test`](https://www.kaggle.com/datasets/sarmentor/big-5-personality-test).

- ~1.015.342 respostas, coletadas 2016–2018 via questionário online.
- Separador: **TAB**. Arquivo: `IPIP-FFM-data-8Nov2018/data-final.csv`.
- 50 itens Likert 1–5 nos blocos `EXT1..EXT10`, `EST1..EST10`, `AGR1..AGR10`,
  `CSN1..CSN10`, `OPN1..OPN10` (O C E A N; `EST` = Neuroticism).
- Colunas `*_E` = tempo de resposta por item (ms); há também metadados
  (`dateload`, `screenw/h`, `country`, `IPC`, `lat_appx_lots_of_err`, …).
- `0` num item = não respondido → tratado como `NaN` no loader.

## Como baixar

```bash
bash scripts/get_data.sh      # baixa e extrai em data/raw/IPIP-FFM-data-8Nov2018/
```

## Chave de itens reversos (IPIP-50 padrão)

Aplicada por `src/fuzzyprofile/data.py::reverse_score` — `x' = (x_max + x_min) − x = 6 − x`:

| Fator | Itens reversos (1-indexados no bloco) |
|-------|--------------------------------------|
| EXT   | 2, 4, 6, 8, 10 |
| EST   | 2, 4 |
| AGR   | 1, 3, 5, 7 |
| CSN   | 2, 4, 6, 8 |
| OPN   | 2, 4, 6 |

## Sem rede / execução imediata

`src/fuzzyprofile/data.py::make_synthetic_ipip()` gera respostas sintéticas a partir de
`c` perfis OCEAN latentes com ruído ordinal e polaridade por item — o pipeline roda sem
nenhum download.

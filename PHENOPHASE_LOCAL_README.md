# Módulo Phenophase Local - Guia de Implementação

## 📋 O que foi criado

Implementação local de extração de estágios fenológicos (SOS, POS, EOS) para séries temporais de NDVI, substituindo a dependência do wcpms do INPE.

### Arquivos Criados:

1. **`phenophase_local.py`** ⭐ 
   - Módulo principal com todas as funções
   - ~500 linhas de código bem documentado
   - Totalmente independente de serviços web

2. **`test_phenophase_local.py`**
   - Script de teste completo com dados sintéticos
   - Gera visualizações de ciclos e sazonalidade
   - Exemplo de uso prático

3. **`exemplos_phenophase_local.py`**
   - 7 exemplos práticos diferentes
   - Demonstra parametrizações por cultura
   - Análise de sensibilidade

4. **`PHENOPHASE_LOCAL_DOCS.md`**
   - Documentação completa
   - Referência de API
   - Troubleshooting

### Arquivo Modificado:

5. **`test_wtss.py`**
   - Removido import de `wcpms`
   - Integrado novo módulo `phenophase_local`
   - Função `get_and_plot_areal_ts_wtss` agora usa método local

## 🎯 Principais Funcionalidades

### 1. Extração de Sazonalidade (STD)
```python
seasonality = extract_seasonality_std(df_ts, window=365)
# Retorna: STD móvel, média móvel, NDVI normalizado
```

### 2. Detecção de Ciclos (Mínimos de NDVI)
```python
cycles = identify_crop_cycles(df_ts, min_cycle_length=60)
# Retorna: Lista com datas e índices de cada ciclo
```

### 3. Ajuste de Gaussiana
```python
fitted = fit_gaussian_to_cycle(df_ts, cycle)
# Retorna: Parâmetros (amplitude, σ, offset) + SOS/POS/EOS
```

### 4. Extração Completa
```python
phenometrics = extract_phenometrics_local(df_ts)
print_phenometrics_summary(phenometrics)
# Retorna: Resumo de todos os ciclos com métricas
```

## 📊 Modelo Matemático

### Gaussiana Ajustada
```
f(x) = A · exp(-(x - μ)² / 2σ²) + C

Onde:
- A = amplitude (diferença max-min)
- μ = mean = POS (Peak of Season)
- σ = std dev (largura do ciclo)
- C = offset (NDVI mínimo)
```

### Estágios Fenológicos (em 25% da amplitude)
```
SOS = μ - √(2σ² · ln(4))    (Start of Season)
POS = μ                       (Peak of Season)
EOS = μ + √(2σ² · ln(4))    (End of Season)
```

## 🚀 Como Usar

### Uso Simples:
```python
from phenophase_local import extract_phenometrics_local, print_phenometrics_summary

# Seus dados devem ter: 'datetime' e 'NDVI_mean'
phenometrics = extract_phenometrics_local(df_ts)
print_phenometrics_summary(phenometrics)
```

### Uso Avançado:
```python
# Customizar para sua cultura
phenometrics = extract_phenometrics_local(
    df_ts,
    ndvi_column='NDVI_mean',
    min_cycle_length=100,        # Ciclo mínimo em dias
    min_ndvi_threshold=0.22      # Limiar para solo exposto
)

# Acessar resultados
for cycle in phenometrics['cycles']:
    if cycle['fit_success']:
        print(f"SOS: {cycle['phenophase_dates']['sos']}")
        print(f"POS: {cycle['phenophase_dates']['pos']}")
        print(f"EOS: {cycle['phenophase_dates']['eos']}")
```

## ✅ Testes Inclusos

### Executar teste completo:
```bash
python test_phenophase_local.py
```

Gera:
- Gráficos de série temporal com ciclos detectados
- Gráficos individuais de cada ciclo com gaussiana ajustada
- Gráfico de sazonalidade (STD móvel)

### Executar exemplos:
```bash
python exemplos_phenophase_local.py
```

Demonstra 7 casos de uso diferentes

## 📈 Resultado Esperado

### Saída do `print_phenometrics_summary()`:
```
================================================================================
RESUMO DE MÉTRICAS FENOLÓGICAS (MÉTODO LOCAL)
================================================================================

Ciclos detectados: 2
Ajustes bem-sucedidos: 2
R² médio: 0.8451
Comprimento médio do ciclo: 368.0 dias

📊 CICLO 1:
   Período: 2020-08-28 a 2021-08-31
   Duração: 368 dias
   R² do ajuste: 0.8451
   📈 Parâmetros da Gaussiana:
      - Amplitude: 0.6901
      - Desvio Padrão: 51.4 dias
      - Offset: 0.1560
   🌱 Estágios Fenológicos:
      - SOS (Start of Season): 2020-10-21 (NDVI=0.3285)
      - POS (Peak of Season):  2021-01-15 (NDVI=0.8461)
      - EOS (End of Season):   2021-04-10 (NDVI=0.3285)
```

## 🔧 Parametrizações Recomendadas

### Soja:
```python
extract_phenometrics_local(
    df_ts,
    min_cycle_length=100,
    min_ndvi_threshold=0.22
)
```

### Milho:
```python
extract_phenometrics_local(
    df_ts,
    min_cycle_length=120,
    min_ndvi_threshold=0.25
)
```

### Café (perene):
```python
extract_phenometrics_local(
    df_ts,
    min_cycle_length=60,
    min_ndvi_threshold=0.35
)
```

## 📊 Vantagens sobre wcpms

| Aspecto | phenophase_local | wcpms |
|---------|------------------|-------|
| Dependência | ✅ Local | ❌ Web service |
| Limite | ✅ Nenhum | ❌ 350 obs |
| Parâmetros | ✅ Quantitativos | ❌ Qualitativos |
| Velocidade | ✅ Instantânea | ❌ Latência |
| Confiabilidade | ✅ 100% | ❌ Depende servidor |
| Customização | ✅ Alta | ❌ Baixa |

## 🎓 Interpretação dos Parâmetros

### Amplitude (A)
- **Alto (>0.6)**: Cultura bem desenvolvida, muita biomassa
- **Médio (0.4-0.6)**: Desenvolvimento normal
- **Baixo (<0.4)**: Desenvolvimento limitado, estresse

### Desvio Padrão (σ)
- **Grande (>60 dias)**: Ciclo vegetativo longo
- **Normal (40-60 dias)**: Ciclo padrão
- **Pequeno (<40 dias)**: Ciclo curto, clima desfavorável

### R² do Ajuste
- **>0.85**: Excelente, ciclo bem definido
- **0.75-0.85**: Bom, ajuste confiável
- **0.60-0.75**: Razoável, verificar dados
- **<0.60**: Ruim, possível problema

## 🛠️ Troubleshooting

### Problema: "Nenhum ciclo detectado"
**Solução:**
```python
# Reduzir min_cycle_length
extract_phenometrics_local(df_ts, min_cycle_length=40)

# Ou aumentar a série para 2+ anos
df_ts = df_ts[df_ts['datetime'].dt.year >= 2020]  # Mínimo 2 anos
```

### Problema: "Ciclos duplicados ou sobrepostos"
**Solução:**
```python
# Aumentar min_cycle_length
extract_phenometrics_local(df_ts, min_cycle_length=120)

# Ou ajustar threshold de NDVI
extract_phenometrics_local(df_ts, min_ndvi_threshold=0.30)
```

### Problema: "Ajuste com baixo R²"
**Solução:**
```python
# Suavizar dados antes
from scipy.signal import savgol_filter
df_ts['NDVI_mean'] = savgol_filter(df_ts['NDVI_mean'], 11, 2)

extract_phenometrics_local(df_ts)
```

## 📚 Próximas Melhorias Possíveis

1. **Modelo duplo de Lorentz**: Para captar ciclos assimétricos
2. **Detecção automática de cultura**: Baseada em parâmetros
3. **API REST local**: Para integração com ferramentas Python/R
4. **Cache de resultados**: Para séries grandes
5. **Validação cruzada**: Comparar com campo verdade

## 📝 Nota sobre Qualidade

Os resultados são tão bons quanto os dados de entrada:
- ✅ Série longa (2+ anos) = resultados melhores
- ✅ Dados sem gaps = melhor detecção de ciclos
- ✅ NDVI suavizado = ajuste mais robusto
- ⚠️ Muito ruído = considerar suavização

## 🤝 Integração com seu workflow

Seu código atual em `test_wtss.py` já foi atualizado:

```python
# ANTES (removido):
from wcpms import *
# ... chamadas complexas ao wcpms

# DEPOIS (novo):
from phenophase_local import extract_phenometrics_local, print_phenometrics_summary

phenometrics = extract_phenometrics_local(df_ts)
if phenometrics.get('success', False):
    # Usar os ciclos detectados
```

Nenhuma mudança necessária na sua chamada de `get_and_plot_areal_ts_wtss()`!

## 📖 Referências

- Documentação completa: `PHENOPHASE_LOCAL_DOCS.md`
- Exemplos práticos: `exemplos_phenophase_local.py`
- Testes: `test_phenophase_local.py`

---

**Versão**: 1.0  
**Data**: Janeiro 2026  
**Autor**: Implementação Local

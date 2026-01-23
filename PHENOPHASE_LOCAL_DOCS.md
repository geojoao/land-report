# Módulo de Extração Local de Estágios Fenológicos (phenophase_local.py)

## 📋 Resumo

Substituição local para o serviço wcpms do INPE para extração de estágios fenológicos (SOS, POS, EOS) de séries temporais de NDVI. O módulo utiliza três técnicas principais:

1. **Extração de Sazonalidade**: Desvio padrão móvel (STD) para identificar padrões sazonais
2. **Detecção de Ciclos**: Identificação de mínimos locais de NDVI para separar safras individuais
3. **Ajuste Gaussiano**: Modelagem de cada ciclo com função gaussiana para extrair parâmetros fenológicos

## 🎯 Vantagens

- ✅ **Independente**: Sem dependência de serviços web externos
- ✅ **Rápido**: Processamento local, sem latência de rede
- ✅ **Paramétrico**: Extrai parâmetros quantitativos da gaussiana
- ✅ **Robusto**: Detecta múltiplos ciclos de safra automaticamente
- ✅ **Flexível**: Parâmetros ajustáveis para diferentes culturas e regiões

## 📦 Funcionalidades Principais

### 1. `gaussian(x, amplitude, mean, std, offset)`
**Modelo matemático da gaussiana**

```
f(x) = amplitude * exp(-(x - mean)² / (2 * std²)) + offset
```

Parâmetros:
- `amplitude`: Altura da gaussiana
- `mean`: Centro (corresponde ao POS)
- `std`: Desvio padrão (largura)
- `offset`: Deslocamento vertical (baseline)

### 2. `extract_seasonality_std(df_ts, ndvi_column, window)`
**Extrai sazonalidade usando desvio padrão móvel**

```python
from phenophase_local import extract_seasonality_std
import pandas as pd

# df_ts deve ter colunas 'datetime' e 'NDVI_mean'
seasonality = extract_seasonality_std(df_ts, ndvi_column='NDVI_mean', window=365)

# Retorna dicionário com:
# - datetime: array de timestamps
# - ndvi: valores originais de NDVI
# - std_rolling: desvio padrão móvel
# - mean_rolling: média móvel
# - ndvi_normalized: NDVI normalizado
```

**O que faz:**
- Calcula STD móvel (janela de 365 dias por padrão)
- Remove tendências de longo prazo
- Normaliza valores para remover variações de amplitude

### 3. `identify_crop_cycles(df_ts, ndvi_column, min_ndvi_threshold, prominence, min_cycle_length)`
**Identifica ciclos de safra usando mínimos locais**

```python
from phenophase_local import identify_crop_cycles

cycles = identify_crop_cycles(
    df_ts,
    ndvi_column='NDVI_mean',
    min_ndvi_threshold=None,  # Auto-detecta (20º percentil)
    prominence=0.15,           # Proeminência dos picos
    min_cycle_length=60        # Mínimo de 60 dias
)

# Retorna lista de dicionários:
# [
#   {
#     'cycle_num': 1,
#     'start_idx': 15,
#     'end_idx': 45,
#     'start_date': Timestamp('2020-09-15'),
#     'end_date': Timestamp('2021-08-31'),
#     'length_days': 350.5,
#     'start_ndvi': 0.18,
#     'end_ndvi': 0.16
#   },
#   ...
# ]
```

**O que faz:**
- Suaviza série com filtro Savitzky-Golay
- Detecta mínimos locais de NDVI (solo exposto)
- Agrupa mínimos próximos
- Cria ciclos entre mínimos consecutivos
- Filtra ciclos muito curtos

### 4. `fit_gaussian_to_cycle(df_ts, cycle, ndvi_column)`
**Ajusta gaussiana a um ciclo específico**

```python
from phenophase_local import fit_gaussian_to_cycle

fitted_cycle = fit_gaussian_to_cycle(df_ts, cycle)

# Retorna dicionário com:
# {
#   'cycle_num': 1,
#   'cycle_start': Timestamp('2020-09-15'),
#   'cycle_end': Timestamp('2021-08-31'),
#   'cycle_length_days': 350.5,
#   'fit_success': True,
#   'r_squared': 0.8451,
#   'gaussian_params': {
#     'amplitude': 0.65,
#     'mean_days': 120.5,    # Dias desde início do ciclo
#     'std_dev_days': 51.4,
#     'offset': 0.156
#   },
#   'phenophase_dates': {
#     'sos': Timestamp('2020-10-21'),  # Start of Season
#     'pos': Timestamp('2021-01-15'),  # Peak of Season
#     'eos': Timestamp('2021-04-10')   # End of Season
#   },
#   'phenophase_values': {
#     'sos_ndvi': 0.3285,
#     'pos_ndvi': 0.8461,
#     'eos_ndvi': 0.3285
#   }
# }
```

**O que faz:**
- Extrai dados do ciclo
- Converte datas para dias desde início
- Estima parâmetros iniciais
- Ajusta gaussiana com otimização não-linear
- Calcula R² da qualidade do ajuste
- Extrai SOS/POS/EOS em 25% da amplitude

### 5. `extract_phenometrics_local(df_ts, ndvi_column, min_cycle_length, min_ndvi_threshold)`
**Função principal: extração completa de fenometria**

```python
from phenophase_local import extract_phenometrics_local, print_phenometrics_summary

phenometrics = extract_phenometrics_local(
    df_ts,
    ndvi_column='NDVI_mean',
    min_cycle_length=60,
    min_ndvi_threshold=None
)

# Imprime resumo formatado
print_phenometrics_summary(phenometrics)

# Acessa resultados:
print(f"Ciclos detectados: {phenometrics['num_cycles']}")
print(f"Ciclos com ajuste bem-sucedido: {phenometrics['num_successful_fits']}")
print(f"R² médio: {phenometrics['mean_r_squared']:.4f}")

# Processa cada ciclo
for cycle in phenometrics['cycles']:
    if cycle['fit_success']:
        print(f"Ciclo {cycle['cycle_num']}: {cycle['phenophase_dates']['pos']}")
```

## 📊 Exemplo de Uso Completo

```python
import pandas as pd
from datetime import datetime, timedelta
from phenophase_local import extract_phenometrics_local, print_phenometrics_summary

# 1. Carrega dados de série temporal NDVI
# Esperado: DataFrame com colunas 'datetime' e 'NDVI_mean'
df_ts = pd.read_csv('ndvi_timeseries.csv')
df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])

# 2. Extrai fenometria completa
phenometrics = extract_phenometrics_local(
    df_ts,
    ndvi_column='NDVI_mean',
    min_cycle_length=60,        # Ciclo mínimo de 60 dias
    min_ndvi_threshold=0.25     # Limiar para solo exposto
)

# 3. Imprime resumo
print_phenometrics_summary(phenometrics)

# 4. Usa os resultados
for cycle in phenometrics['cycles']:
    if cycle['fit_success']:
        params = cycle['gaussian_params']
        phases = cycle['phenophase_dates']
        
        print(f"\nCiclo {cycle['cycle_num']}:")
        print(f"  Duração: {cycle['cycle_length_days']:.0f} dias")
        print(f"  Amplitude NDVI: {params['amplitude']:.3f}")
        print(f"  Desvio Padrão: {params['std_dev_days']:.1f} dias")
        print(f"  SOS: {phases['sos'].strftime('%Y-%m-%d')}")
        print(f"  POS: {phases['pos'].strftime('%Y-%m-%d')}")
        print(f"  EOS: {phases['eos'].strftime('%Y-%m-%d')}")
        print(f"  R²: {cycle['r_squared']:.4f}")
```

## 🔧 Integração com o código existente

No arquivo `test_wtss.py`, a função `get_and_plot_areal_ts_wtss` foi atualizada para usar o novo módulo:

```python
from phenophase_local import extract_phenometrics_local, print_phenometrics_summary

# Na função:
phenometrics = extract_phenometrics_local(df_ts, ndvi_column='NDVI_mean')

if phenometrics.get('success', False) and phenometrics['cycles']:
    successful_cycles = [c for c in phenometrics['cycles'] if c.get('fit_success', False)]
    if successful_cycles:
        # Usa o primeiro ciclo bem-sucedido
        primary_cycle = successful_cycles[0]
        phenophase = primary_cycle['phenophase_dates']
        # ... plota SOS, POS, EOS
```

## 📈 Definição dos Estágios Fenológicos

- **SOS (Start of Season)**: Data onde o NDVI atinge 25% da amplitude desde o offset
  - Marca o início do desenvolvimento vegetativo
  - Aproximadamente no estádio V4-V5 em soja

- **POS (Peak of Season)**: Data do máximo valor de NDVI
  - Máximo desenvolvimento foliar
  - Aproximadamente no estádio R2-R3 em soja

- **EOS (End of Season)**: Data onde o NDVI retorna a 25% da amplitude
  - Fim do ciclo vegetativo
  - Aproximadamente na colheita

## 🛠️ Parâmetros Ajustáveis

### Para `identify_crop_cycles`:
- `min_cycle_length`: Comprimento mínimo de ciclo (padrão: 60 dias)
  - Aumentar para culturas com ciclos mais longos
  - Diminuir para detectar ciclos mais curtos
  
- `min_ndvi_threshold`: Limiar para detectar solo exposto (padrão: auto)
  - `None`: Auto-detecta no 20º percentil
  - Valor fixo: 0.2-0.3 para solo exposto típico

- `prominence`: Proeminência para detecção de picos (padrão: 0.15)
  - Aumentar para menos ciclos detectados (menos sensível)
  - Diminuir para mais ciclos detectados (mais sensível)

### Para `extract_seasonality_std`:
- `window`: Tamanho da janela móvel em dias (padrão: 365)
  - 365 dias: Captura variação sazonal anual
  - Diminuir para capturar ciclos mais curtos

## 📊 Interpretação dos Resultados

### R² (Coeficiente de Determinação)
- **R² > 0.85**: Excelente ajuste
- **R² > 0.75**: Bom ajuste
- **R² > 0.60**: Ajuste razoável
- **R² < 0.60**: Ajuste ruim (verificar ciclo)

### Amplitude
- Indica a diferença entre NDVI mínimo e máximo
- Culturas com maior biomassa têm amplitude maior
- Valores típicos: 0.4-0.8 para culturas bem desenvolvidas

### Desvio Padrão
- Indica a largura do ciclo vegetativo
- Culturas com ciclo longo: STD > 60 dias
- Culturas com ciclo curto: STD < 40 dias

## 🐛 Troubleshooting

### "Nenhum ciclo detectado"
1. Verifique se a série tem pelo menos 2 anos de dados
2. Aumente o valor de `min_cycle_length`
3. Diminua o limiar `min_ndvi_threshold`
4. Reduza a `prominence`

### "Ajuste com baixo R²"
1. Verifique se há ruído excessivo nos dados
2. Suavize a série antes de passar para a função
3. Ajuste os parâmetros de detecção de ciclos
4. Verifique se o ciclo é realmente sazonal

### "SOS/POS/EOS em datas incoerentes"
1. Verifique os limites do ciclo detectado
2. Aumente a `prominence` para melhor detecção de ciclos
3. Use `min_ndvi_threshold` mais agressivo

## 📚 Referências Científicas

A abordagem gaussiana é baseada em:
- Fitting Gaussian functions to phenological curves for better parameter interpretation
- Usando STD para detecção de sazonalidade é método padrão em análise de séries temporais
- Detecção de mínimos para separação de ciclos: método clássico em fenologia agrícola

## 🔄 Comparação com wcpms

| Aspecto | phenophase_local | wcpms (INPE) |
|---------|------------------|--------------|
| Dependência | Local | Web service |
| Latência | Mínima | Variável |
| Limite de dados | Nenhum | 350 observações |
| Parâmetros | Quantitativos (gaussiana) | Qualitativo (thresholds) |
| Customização | Alta | Baixa |
| Velocidade | Muito rápida | Lenta |
| Confiabilidade | 100% (local) | Depende do servidor |


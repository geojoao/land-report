# Plano de Testes - Modificações no bocom_bbm_report.qmd

Data: 28 de janeiro de 2026
Status: ✅ Implementadas todas as 5 modificações

---

## Teste 1: Cálculo de Volume Total da Operação

### Objetivo
Verificar se o volume total é calculado corretamente e exibido com formatação adequada.

### Procedimento
1. Abrir XML de teste com múltiplas operações
2. Verificar se `VlrTotOp` está sendo somado
3. Confirmar formatação `R$ X.XXX.XXX,XX`

### Dados de Teste
```python
# Simular dados
gdf['VlrTotOp'] = [10000000.0, 20000000.0, 8000000.0]
# Esperado: "R$ 38.000.000,00"
```

### Critérios de Sucesso
- [ ] Volume total exibido na seção "Dados Cadastrais"
- [ ] Formatação: `R$ X.XXX.XXX,XX` (com separador de milhar e decimal corretos)
- [ ] Sem erros Python durante cálculo
- [ ] Trata corretamente valores ausentes (exibe 'N/A')

### Comando para Testar
```bash
quarto render bocom_bbm_report.qmd --execute-params '{"produtor": "TesteProd", "ref_bacen": "123456789"}'
```

---

## Teste 2: Formatação Numérica com Unidades

### Objetivo
Validar que todos os campos numéricos exibem unidades de medida corretas e formatação locale-aware.

### Campos a Verificar

#### Dados Cadastrais (seção 1)
```
Área Financiada (ha): X.XXX,XX ha
Volume Total (BRL): R$ X.XXX.XXX,XX
Receita Esperada (BRL): R$ X.XXX,XX
```

#### Análise de Desvios (seção 2)
```
Área do cultivo (ha): X.XXX,XX ha
Expectativa de produtividade: X.XXX,XX kg/ha
Expectativa de produção: X.XXX,XX kg
```

### Procedimento de Teste
```python
# Testar formatação
import pandas as pd

area_ha = 1500.5678
receita = 38000000.0
produtividade = 7500.123

print(f"Área: {area_ha:,.2f} ha")  # Esperado: "Área: 1.500,57 ha"
print(f"Receita: R$ {receita:,.2f}")  # Esperado: "Receita: R$ 38.000.000,00"
print(f"Produtividade: {produtividade:,.2f} kg/ha")  # Esperado: "Produtividade: 7.500,12 kg/ha"
```

### Critérios de Sucesso
- [ ] Separadores corretos (ponto para milhar, vírgula para decimal)
- [ ] Unidades presentes em todos os campos numéricos
- [ ] Valores monetários com símbolo `R$`
- [ ] Áreas com unidade `ha`
- [ ] Produção com unidade `kg` ou `kg/ha`

### Observação
Depende da locale do sistema operacional. Se não exibir corretamente, adicionar:
```python
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
```

---

## Teste 3: Clipping RGB - Normalização de Contraste

### Objetivo
Verificar se as imagens RGB do Sentinel-2 estão com contraste melhorado e sem oversaturação.

### Procedimento Visual
1. Gerar relatório com dados Sentinel disponíveis
2. Comparar visualmente imagens RGB:
   - Antes: Muito brilhantes/saturadas
   - Depois: Contraste adequado, cores naturais

### Procedimento Técnico
```python
# Testar função isoladamente
import numpy as np

# Mock data
b08_raw = np.random.randint(0, 4095, size=(100, 100)).astype(np.float32)  # NIR bruto

# Aplicar clipping
def normalize_and_clip_band(band_data, L, clip_percentile=2):
    # ... código da função ...
    return clipped_band

result = normalize_and_clip_band(b08_raw, L=4096, clip_percentile=2)

# Validações
assert result.min() >= 0, "Min value deve ser >= 0"
assert result.max() <= 1, "Max value deve ser <= 1"
assert not np.isnan(result).any(), "Não deve haver NaNs"
assert result.dtype == np.float32, "Dtype deve ser float32"
```

### Critérios de Sucesso
- [ ] Função `normalize_and_clip_band()` definida corretamente
- [ ] Valores de saída entre 0-1
- [ ] Sem NaNs na saída
- [ ] Percentis de clipping aplicados (2-98%)
- [ ] Imagens RGB visualmente melhoradas
- [ ] Código executa sem exceções

### Verificação de Erros Comuns
```python
# ❌ ERRADO: Sem validação de NaNs
rgb[np.isnan(rgb)] = 0  # Perder informação

# ✅ CORRETO: Usar nan_to_num
rgb = np.nan_to_num(rgb, 0)
```

---

## Teste 4: Filtragem por Estágios Fenológicos

### Objetivo
Verificar que apenas 2-6 imagens por gleba são exibidas, selecionadas estrategicamente por fenofase.

### Procedimento
1. Contar número de imagens Sentinel no catálogo
2. Gerar relatório
3. Contar imagens exibidas
4. Verificar se 2-6 imagens aparecem

### Teste Técnico
```python
# Mock: 30 imagens disponíveis
asset_list = [
    (f'2024-{i:02d}-01', {'rgb': np.random.rand(10, 10, 3), 'evi': np.random.rand(10, 10)})
    for i in range(1, 31)
]

filtered = filter_by_phenophase(asset_list, min_images=2, max_images=6)

# Validações
assert 2 <= len(filtered) <= 6, f"Deve ter 2-6 imagens, tem {len(filtered)}"
assert all(isinstance(item, tuple) and len(item) == 2 for item in filtered), "Formato inválido"
assert filtered[0][0] < filtered[-1][0], "Deve estar ordenado temporalmente"
```

### Critérios de Sucesso
- [ ] Função `filter_by_phenophase()` definida
- [ ] Retorna 2-6 imagens
- [ ] Incluir primeira e última data
- [ ] Incluir data do meio
- [ ] Incluir picos de EVI
- [ ] Manter ordem temporal
- [ ] Log informativo no output: `"Exibindo X de Y imagens"`
- [ ] Nenhuma exceção durante execução

### Validação de Cobertura Temporal
```python
# Verificar se imagens cobrem o período inteiro
dates = [item[0] for item in filtered]
time_span = (dates[-1] - dates[0]).days
assert time_span > 30, "Deve cobrir período mínimo de 30 dias"
```

---

## Teste 5: Robustez XML - Tratamento de Datas

### Objetivo
Garantir que o relatório funcione com datas de plantio preenchidas, ausentes ou inválidas.

### Casos de Teste

#### Caso 5.1: Datas Válidas
```python
gleba_data = {
    'DtEms': pd.Timestamp('2024-01-15'),
    'DtIniPlant': pd.Timestamp('2024-02-01'),
    'DtFimPlant': pd.Timestamp('2024-03-01'),
    'DtIniColht': pd.Timestamp('2024-06-01'),
    'DtFimColht': pd.Timestamp('2024-07-01'),
}
# Esperado: Usar DtIniPlant - DtFimColht
```

#### Caso 5.2: Plantio Ausente
```python
gleba_data = {
    'DtEms': pd.Timestamp('2024-01-15'),
    'DtIniPlant': None,
    'DtFimPlant': None,
    'DtIniColht': None,
    'DtFimColht': None,
}
# Esperado: Usar DtEms até DtEms+1ano
```

#### Caso 5.3: Data Inválida (String)
```python
gleba_data = {
    'DtEms': pd.Timestamp('2024-01-15'),
    'DtIniPlant': '2024-13-01',  # Mês inválido
    'DtFimColht': 'N/A',  # Literal
}
# Esperado: Tratar como NaT, usar fallback
```

### Procedimento de Teste
```python
# Executar função processar_gleba com cada caso
for case in [caso_5_1, caso_5_2, caso_5_3]:
    try:
        processar_gleba(case, gleba_id=1)
        print(f"✅ {case} passou")
    except Exception as e:
        print(f"❌ {case} falhou: {e}")
```

### Critérios de Sucesso
- [ ] Caso 5.1: Utiliza datas de plantio/colheita
- [ ] Caso 5.2: Utiliza fallback DtEms+1ano
- [ ] Caso 5.3: Converte inválidos para NaT, usa fallback
- [ ] Nenhuma exceção `TypeError` ou `ValueError` não tratada
- [ ] Mensagens de status informativas

### Validação de Log
```python
# Verificar que as mensagens aparecem no stderr
# Esperado para caso 5.2/5.3:
# "AVISO: Datas ausentes — processamento do cubo para SOM pulado."
```

### Teste de Integração Completo
```bash
# Teste com 3 XMLs diferentes
python -c "
import subprocess
xmls = ['valid_dates.xml', 'missing_dates.xml', 'invalid_dates.xml']
for xml in xmls:
    result = subprocess.run(
        ['quarto', 'render', 'bocom_bbm_report.qmd'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f'✅ {xml} passou')
    else:
        print(f'❌ {xml} falhou: {result.stderr}')
"
```

---

## Teste 6: Integração Completa

### Objetivo
Executar o relatório completo com dados reais e verificar todos os componentes.

### Procedimento
```bash
# 1. Preparar ambiente
source /workspaces/land-report/.venv/bin/activate

# 2. Usar XML de teste
export XML_FILE="20250862734_Monte Alegre.xml"

# 3. Renderizar
quarto render bocom_bbm_report.qmd --execute-params '{
    "produtor": "Monte Alegre",
    "ref_bacen": "20250862734"
}'

# 4. Verificar output
ls -lh bocom_bbm_report.pdf
```

### Checklist Final
- [ ] Relatório gerado sem erro
- [ ] PDF criado com sucesso
- [ ] Tamanho < 200MB (indica compressão funcionando)
- [ ] Nenhuma página em branco
- [ ] Tabelas formatadas corretamente
- [ ] Imagens RGB visíveis
- [ ] Gráficos WTSS renderizados
- [ ] Todos os dados cadastrais presentes
- [ ] Sem avisos "undefined" em nenhuma tabela

---

## Teste 7: Validação de Dados Cadastrais

### Objetivo
Verificar se todos os campos da seção "Dados Cadastrais" estão presentes e formatados.

### Campos Esperados
```
✓ Instituição Financeira: BOCOM BBM
✓ Produtor: [nome do produtor]
✓ CPF / CNPJ: [CNPJ beneficiário]
✓ REFBACEN: [ref_bacen do XML]
✓ Área Financiada (ha): X.XXX,XX ha
✓ Volume Total da Operação (BRL): R$ X.XXX.XXX,XX  ← NOVO
✓ Receita esperada (BRL): R$ X.XXX,XX
✓ Data de Emissão: [data em ISO]
```

### Validação
```python
# Ler PDF e verificar presença de texto
import pdfplumber

with pdfplumber.open('bocom_bbm_report.pdf') as pdf:
    text = ''.join([page.extract_text() for page in pdf.pages[:3]])
    
    required_fields = [
        'Volume Total da Operação',
        'Receita esperada (BRL)',
        'Área Financiada (ha)',
        'R$'
    ]
    
    for field in required_fields:
        if field in text:
            print(f"✅ {field} encontrado")
        else:
            print(f"❌ {field} NÃO encontrado")
```

---

## Teste 8: Performance e Tamanho

### Objetivo
Garantir que otimizações implementadas realmente reduzem tamanho do PDF.

### Medidas
```bash
# Antes (esperado ~200MB com todas as imagens)
# Depois (esperado ~60-100MB com filtragem)

du -h bocom_bbm_report.pdf
wc -l bocom_bbm_report.qmd  # ~2155 linhas
```

### Benchmark
| Operação | Tempo |
|----------|-------|
| XML Parse | < 1s |
| Dados Cadastrais | < 1s |
| Mapa | 2-5s |
| WTSS | 3-8s |
| Sentinel Search | 5-15s |
| Sentinel Processing | 20-50s |
| RGB Clipping | < 2s |
| Filtragem Fenofase | 1-3s |
| SOM | 20-40s |
| **Total** | **< 3 minutos** |

---

## Matriz de Rastreabilidade

| Modificação | Teste | Critério | Status |
|-------------|-------|----------|--------|
| Volume Total | 1 | Soma VlrTotOp corretamente | ⏳ |
| Formatação Numérica | 2 | Unidades e separadores corretos | ⏳ |
| Clipping RGB | 3 | Função implementada e executa | ⏳ |
| Filtragem Fenofase | 4 | 2-6 imagens selecionadas | ⏳ |
| Robustez XML | 5 | Trata datas inválidas | ⏳ |
| Integração | 6 | PDF gerado completo | ⏳ |
| Validação Dados | 7 | Todos os campos presentes | ⏳ |
| Performance | 8 | Tamanho < 150MB | ⏳ |

**Legenda:** ⏳ Pendente | ✅ Passou | ❌ Falhou

---

## Próximas Etapas

1. [ ] Executar os 8 testes acima
2. [ ] Documentar qualquer desvio encontrado
3. [ ] Ajustar parâmetros se necessário (max_images, clip_percentile, etc.)
4. [ ] Validar em ambiente de produção
5. [ ] Treinar usuários nas novas funcionalidades

---

## Suporte

Para questões sobre os testes ou modificações, ver:
- [CHANGES_SUMMARY.md](./CHANGES_SUMMARY.md) - Resumo das mudanças
- [TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md) - Documentação técnica
- [bocom_bbm_report.qmd](./bocom_bbm_report.qmd) - Código-fonte

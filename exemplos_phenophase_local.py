"""
Exemplos práticos de uso do módulo phenophase_local.py
Demonstra diferentes cenários e parametrizações
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from phenophase_local import (
    extract_phenometrics_local,
    print_phenometrics_summary,
    identify_crop_cycles,
    extract_seasonality_std,
    gaussian
)


# ============================================================================
# EXEMPLO 1: Série real com dados WTSS
# ============================================================================
def exemplo_1_dados_reais():
    """
    Como usar com dados reais do WTSS.
    """
    print("\n" + "="*80)
    print("EXEMPLO 1: Usando dados reais do WTSS")
    print("="*80)
    
    print("""
    Passos:
    1. Obter dados do WTSS (já está em test_wtss.py)
    2. Certificar que tem colunas 'datetime' e 'NDVI_mean'
    3. Chamar extract_phenometrics_local()
    4. Visualizar resultados com print_phenometrics_summary()
    
    Exemplo de código:
    ```python
    from phenophase_local import extract_phenometrics_local, print_phenometrics_summary
    
    phenometrics = extract_phenometrics_local(df_ts)
    print_phenometrics_summary(phenometrics)
    ```
    """)


# ============================================================================
# EXEMPLO 2: Ajuste fino de parâmetros
# ============================================================================
def exemplo_2_parametrizacao(df_ts):
    """
    Como ajustar parâmetros para diferentes culturas.
    """
    print("\n" + "="*80)
    print("EXEMPLO 2: Parametrizações para diferentes culturas")
    print("="*80)
    
    # SOJA (Ciclo típico: 120-150 dias)
    print("\n🌱 SOJA:")
    phenometrics_soja = extract_phenometrics_local(
        df_ts,
        ndvi_column='NDVI_mean',
        min_cycle_length=100,        # Ciclo mínimo de 100 dias
        min_ndvi_threshold=0.22      # Limiar para solo exposto
    )
    print(f"   Ciclos detectados: {phenometrics_soja['num_cycles']}")
    print(f"   Ciclo médio: {phenometrics_soja['mean_cycle_length_days']:.0f} dias")
    
    # MILHO (Ciclo típico: 140-160 dias)
    print("\n🌽 MILHO:")
    phenometrics_milho = extract_phenometrics_local(
        df_ts,
        ndvi_column='NDVI_mean',
        min_cycle_length=120,
        min_ndvi_threshold=0.25
    )
    print(f"   Ciclos detectados: {phenometrics_milho['num_cycles']}")
    print(f"   Ciclo médio: {phenometrics_milho['mean_cycle_length_days']:.0f} dias")
    
    # CAFÉ (Ciclo perene, vários fluxos)
    print("\n☕ CAFÉ:")
    phenometrics_cafe = extract_phenometrics_local(
        df_ts,
        ndvi_column='NDVI_mean',
        min_cycle_length=60,         # Ciclos mais curtos
        min_ndvi_threshold=0.35      # Nunca tem solo muito exposto
    )
    print(f"   Ciclos detectados: {phenometrics_cafe['num_cycles']}")
    print(f"   Ciclo médio: {phenometrics_cafe['mean_cycle_length_days']:.0f} dias")


# ============================================================================
# EXEMPLO 3: Análise de sazonalidade
# ============================================================================
def exemplo_3_sazonalidade(df_ts):
    """
    Como usar apenas a extração de sazonalidade.
    """
    print("\n" + "="*80)
    print("EXEMPLO 3: Análise de Sazonalidade (STD móvel)")
    print("="*80)
    
    seasonality = extract_seasonality_std(
        df_ts,
        ndvi_column='NDVI_mean',
        window=180  # Reduzido para série de teste (era 365)
    )
    
    std_values = seasonality['std_rolling']
    valid_std = std_values[~np.isnan(std_values)]
    
    if len(valid_std) > 0:
        print(f"\n   STD máximo: {np.nanmax(std_values):.4f}")
        print(f"   STD mínimo: {np.nanmin(std_values):.4f}")
        print(f"   STD médio: {np.nanmean(std_values):.4f}")
        
        # Encontra períodos de máxima variabilidade
        idx_max_std = np.nanargmax(std_values)
        
        dates = pd.to_datetime(seasonality['datetime'])
        print(f"\n   Período de máxima sazonalidade: {dates[idx_max_std]}")
    else:
        print("\n   ⚠️ Série muito curta para calcular STD móvel")
        print("      Use série com 3+ anos para resultados melhores")
        return
    
    # Plota
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates, std_values, alpha=0.3, color='orange')
    ax.plot(dates, std_values, color='orange', linewidth=2, label='STD Móvel (180 dias)')
    ax.set_xlabel('Data')
    ax.set_ylabel('Desvio Padrão')
    ax.set_title('Sazonalidade: Desvio Padrão Móvel do NDVI')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exemplo_sazonalidade.png', dpi=150)
    print("\n   📊 Gráfico salvo: exemplo_sazonalidade.png")


# ============================================================================
# EXEMPLO 4: Detecção de ciclos apenas
# ============================================================================
def exemplo_4_deteccao_ciclos(df_ts):
    """
    Como usar apenas a detecção de ciclos.
    """
    print("\n" + "="*80)
    print("EXEMPLO 4: Detecção de Ciclos de Safra")
    print("="*80)
    
    cycles = identify_crop_cycles(
        df_ts,
        ndvi_column='NDVI_mean',
        min_ndvi_threshold=None,  # Auto-detecta
        prominence=0.15,
        min_cycle_length=60
    )
    
    print(f"\n   Total de ciclos detectados: {len(cycles)}")
    
    for i, cycle in enumerate(cycles, 1):
        print(f"\n   Ciclo {i}:")
        print(f"      Início: {cycle['start_date'].strftime('%Y-%m-%d')}")
        print(f"      Fim:    {cycle['end_date'].strftime('%Y-%m-%d')}")
        print(f"      Duração: {cycle['length_days']:.0f} dias")
        print(f"      NDVI min: {min(cycle['start_ndvi'], cycle['end_ndvi']):.4f}")
        print(f"      NDVI máx (previsto): {df_ts.loc[cycle['start_idx']:cycle['end_idx'], 'NDVI_mean'].max():.4f}")


# ============================================================================
# EXEMPLO 5: Extração de parâmetros fenológicos
# ============================================================================
def exemplo_5_parametros_fenologicos(df_ts):
    """
    Como extrair e usar os parâmetros da gaussiana.
    """
    print("\n" + "="*80)
    print("EXEMPLO 5: Extração de Parâmetros Fenológicos")
    print("="*80)
    
    phenometrics = extract_phenometrics_local(df_ts)
    
    print(f"\nResumo da Fenometria:")
    print(f"   R² médio dos ajustes: {phenometrics['mean_r_squared']:.4f}")
    print(f"   Ciclo médio: {phenometrics['mean_cycle_length_days']:.0f} dias")
    
    # Cria tabela com parâmetros
    print(f"\n{'Ciclo':<6} {'Amplitude':<12} {'STD (dias)':<12} {'R²':<8} {'Duração':<12}")
    print("   " + "-"*50)
    
    for cycle in phenometrics['cycles']:
        if cycle['fit_success']:
            params = cycle['gaussian_params']
            print(f"   {cycle['cycle_num']:<6} "
                  f"{params['amplitude']:<12.4f} "
                  f"{params['std_dev_days']:<12.1f} "
                  f"{cycle['r_squared']:<8.4f} "
                  f"{cycle['cycle_length_days']:<12.0f}")
    
    # Calcula duração do ciclo vegetativo ativo (2 * desvio padrão)
    print(f"\n   Duração média do ciclo ativo (2σ): "
          f"{2 * np.mean([c['gaussian_params']['std_dev_days'] for c in phenometrics['cycles'] if c['fit_success']]):.1f} dias")


# ============================================================================
# EXEMPLO 6: Comparação de diferentes thresholds
# ============================================================================
def exemplo_6_sensibilidade(df_ts):
    """
    Como variar parâmetros para entender sensibilidade.
    """
    print("\n" + "="*80)
    print("EXEMPLO 6: Análise de Sensibilidade")
    print("="*80)
    
    # Testa diferentes valores de min_cycle_length
    print("\n   Variando comprimento mínimo do ciclo:")
    print(f"   {'Min Cycle (dias)':<20} {'Ciclos Detectados':<20} {'R² Médio':<15}")
    print("   " + "-"*55)
    
    for min_len in [30, 60, 90, 120, 150]:
        pheno = extract_phenometrics_local(df_ts, min_cycle_length=min_len)
        mean_r2 = pheno.get('mean_r_squared', 0.0) if pheno.get('success', False) else 0.0
        print(f"   {min_len:<20} {pheno.get('num_cycles', 0):<20} {mean_r2:<15.4f}")
    
    # Testa diferentes thresholds de NDVI
    print("\n   Variando threshold de NDVI mínimo:")
    print(f"   {'NDVI Threshold':<20} {'Ciclos Detectados':<20} {'R² Médio':<15}")
    print("   " + "-"*55)
    
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35]:
        pheno = extract_phenometrics_local(df_ts, min_ndvi_threshold=threshold)
        mean_r2 = pheno.get('mean_r_squared', 0.0) if pheno.get('success', False) else 0.0
        print(f"   {threshold:<20.2f} {pheno.get('num_cycles', 0):<20} {mean_r2:<15.4f}")


# ============================================================================
# EXEMPLO 7: Exportando resultados
# ============================================================================
def exemplo_7_exportar_resultados(phenometrics):
    """
    Como exportar os resultados em diferentes formatos.
    """
    print("\n" + "="*80)
    print("EXEMPLO 7: Exportação de Resultados")
    print("="*80)
    
    # Exportar para CSV
    records = []
    for cycle in phenometrics['cycles']:
        if cycle['fit_success']:
            records.append({
                'cycle_num': cycle['cycle_num'],
                'cycle_start': cycle['cycle_start'],
                'cycle_end': cycle['cycle_end'],
                'cycle_length_days': cycle['cycle_length_days'],
                'amplitude': cycle['gaussian_params']['amplitude'],
                'std_dev_days': cycle['gaussian_params']['std_dev_days'],
                'offset': cycle['gaussian_params']['offset'],
                'r_squared': cycle['r_squared'],
                'sos_date': cycle['phenophase_dates']['sos'],
                'pos_date': cycle['phenophase_dates']['pos'],
                'eos_date': cycle['phenophase_dates']['eos'],
                'sos_ndvi': cycle['phenophase_values']['sos_ndvi'],
                'pos_ndvi': cycle['phenophase_values']['pos_ndvi'],
                'eos_ndvi': cycle['phenophase_values']['eos_ndvi']
            })
    
    df_export = pd.DataFrame(records)
    df_export.to_csv('phenometrics_resultado.csv', index=False)
    print("\n   ✓ Exportado para: phenometrics_resultado.csv")
    
    # JSON
    import json
    
    # Converte timestamps para string para JSON
    phenometrics_json = phenometrics.copy()
    for cycle in phenometrics_json['cycles']:
        if cycle['fit_success']:
            cycle['cycle_start'] = cycle['cycle_start'].isoformat()
            cycle['cycle_end'] = cycle['cycle_end'].isoformat()
            cycle['phenophase_dates'] = {k: v.isoformat() for k, v in cycle['phenophase_dates'].items()}
    
    with open('phenometrics_resultado.json', 'w') as f:
        json.dump(phenometrics_json, f, indent=2, default=str)
    print("   ✓ Exportado para: phenometrics_resultado.json")


# ============================================================================
# Função principal
# ============================================================================
def main():
    """Executa todos os exemplos."""
    
    # Cria dados de teste (SÉ RIES LONGAS!)
    print("\n🔧 Gerando dados de teste...")
    np.random.seed(42)
    
    # IMPORTANTE: Usar série longa (100+ pontos = 3+ anos)
    dates = pd.date_range('2020-01-01', periods=150, freq='16D')
    ndvi = []
    for i, date in enumerate(dates):
        day = (date - dates[0]).days
        day_in_year = day % 365
        
        if day_in_year < 60:
            val = 0.15 + 0.05 * (day_in_year / 60)
        else:
            x = (day_in_year - 120) / 60
            val = 0.2 + 0.65 * np.exp(-x**2 / 2)
        
        ndvi.append(np.clip(val + np.random.normal(0, 0.05), 0.1, 1.0))
    
    df_ts = pd.DataFrame({
        'datetime': dates,
        'NDVI_mean': ndvi
    })
    
    print(f"✓ Dados gerados ({len(df_ts)} observações, {(df_ts['datetime'].max() - df_ts['datetime'].min()).days} dias)\n")
    
    # Executa exemplos
    exemplo_1_dados_reais()
    exemplo_2_parametrizacao(df_ts)
    exemplo_3_sazonalidade(df_ts)
    exemplo_4_deteccao_ciclos(df_ts)
    exemplo_5_parametros_fenologicos(df_ts)
    exemplo_6_sensibilidade(df_ts)
    
    # Exemplo 7 precisa de phenometrics
    phenometrics = extract_phenometrics_local(df_ts)
    exemplo_7_exportar_resultados(phenometrics)
    
    print("\n" + "="*80)
    print("✅ Todos os exemplos executados!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

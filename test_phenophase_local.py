"""
Script de teste e demonstração do módulo phenophase_local.py
Exemplifica o uso de todas as funções para extração de estágios fenológicos.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from phenophase_local import (
    extract_phenometrics_local,
    print_phenometrics_summary,
    identify_crop_cycles,
    fit_gaussian_to_cycle,
    extract_seasonality_std,
    gaussian
)

def create_synthetic_ts_data(num_years=3, cycles_per_year=1, noise_level=0.05):
    """
    Cria dados sintéticos de série temporal NDVI para testes.
    
    Args:
        num_years: Número de anos de dados
        cycles_per_year: Ciclos de safra por ano
        noise_level: Nível de ruído a ser adicionado
    
    Returns:
        DataFrame com dados sintéticos
    """
    dates = []
    ndvi_values = []
    
    start_date = datetime(2020, 1, 1)
    current_date = start_date
    end_date = start_date + timedelta(days=365*num_years)
    
    np.random.seed(42)  # Para reprodutibilidade
    
    # Gera dados com padrão sazonal realista
    day_counter = 0
    while current_date <= end_date:
        # Padrão sazonal de crescimento (gaussiana por ciclo)
        
        # Plantio inicia aproximadamente em setembro (dia 250)
        # Colheita aproximadamente em abril (dia 100 do próximo ano)
        
        if cycles_per_year == 1:
            # Um ciclo por ano (safra anual típica de soja/milho)
            # Ciclo aprox: setembro (250) até agosto (243)
            
            day_in_year = current_date.timetuple().tm_yday
            
            # Ajusta para o ciclo começar em setembro
            if day_in_year >= 250:  # Setembro em diante
                days_in_cycle = day_in_year - 250
            else:  # Janeiro até agosto
                days_in_cycle = 365 - 250 + day_in_year
            
            cycle_length = 365  # 365 dias de ciclo
            
            # Solo exposto (NDVI baixo) nos primeiros 60 dias
            if days_in_cycle < 60:
                ndvi = 0.15 + 0.05 * (days_in_cycle / 60)
            else:
                # Crescimento e pico (dias 60-180)
                # Gaussiana centrada no dia 120 (máximo desenvolvimento)
                x = (days_in_cycle - 120) / 60
                ndvi = 0.2 + 0.65 * np.exp(-x**2 / 2)
        else:
            # Múltiplos ciclos por ano
            cycle_length = 365 / cycles_per_year
            phase = (day_counter % cycle_length) / cycle_length
            ndvi = 0.25 + 0.55 * np.exp(-(phase - 0.5)**2 / (2 * 0.12**2))
        
        # Adiciona ruído gaussiano
        ndvi += np.random.normal(0, noise_level)
        ndvi = np.clip(ndvi, 0.1, 1.0)
        
        dates.append(current_date)
        ndvi_values.append(ndvi)
        
        # Avança 16 dias (frequência MODIS MOD13Q1)
        current_date += timedelta(days=16)
        day_counter += 16
    
    df = pd.DataFrame({
        'datetime': dates,
        'NDVI_mean': ndvi_values
    })
    
    return df


def plot_cycle_with_gaussian(df_ts, cycle, fitted_cycle, save_path=None):
    """
    Plota um ciclo individual com o ajuste gaussiano sobreposto.
    
    Args:
        df_ts: DataFrame com série temporal
        cycle: Dicionário de ciclo
        fitted_cycle: Resultado do ajuste gaussiano
        save_path: Caminho para salvar a figura (opcional)
    """
    df_ts = df_ts.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    df_ts = df_ts.sort_values('datetime')
    
    start_idx = cycle['start_idx']
    end_idx = cycle['end_idx']
    
    cycle_data = df_ts.iloc[start_idx:end_idx + 1].copy()
    ndvi_cycle = cycle_data['NDVI_mean'].values
    dates_cycle = cycle_data['datetime'].values
    
    # Converte datas para dias
    days_since_start = np.array([(d - dates_cycle[0]) / np.timedelta64(1, 'D') for d in dates_cycle])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plota dados reais
    ax.plot(dates_cycle, ndvi_cycle, 'o-', color='#003366', linewidth=2, markersize=4, label='NDVI observado')
    
    # Plota gaussiana ajustada
    if fitted_cycle.get('fit_success', False):
        days_smooth = np.linspace(days_since_start[0], days_since_start[-1], 200)
        dates_smooth = [pd.Timestamp(dates_cycle[0]) + timedelta(days=d) for d in days_smooth]
        
        params = fitted_cycle['gaussian_params']
        ndvi_smooth = gaussian(days_smooth, 
                             amplitude=params['amplitude'],
                             mean=params['mean_days'],
                             std=params['std_dev_days'],
                             offset=params['offset'])
        
        ax.plot(dates_smooth, ndvi_smooth, '--', color='#ff6b00', linewidth=2.5, label='Ajuste Gaussiano')
        
        # Marca SOS, POS, EOS
        phenophase = fitted_cycle['phenophase_dates']
        phenophase_val = fitted_cycle['phenophase_values']
        
        ax.axvline(phenophase['sos'], color='blue', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.plot(phenophase['sos'], phenophase_val['sos_ndvi'], 'o', color='blue', markersize=8, label='SOS')
        
        ax.axvline(phenophase['pos'], color='red', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.plot(phenophase['pos'], phenophase_val['pos_ndvi'], 'o', color='red', markersize=8, label='POS')
        
        ax.axvline(phenophase['eos'], color='purple', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.plot(phenophase['eos'], phenophase_val['eos_ndvi'], 'o', color='purple', markersize=8, label='EOS')
        
        # Adiciona texto com R²
        ax.text(0.98, 0.05, f"R² = {fitted_cycle['r_squared']:.4f}", 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
    
    ax.set_xlabel('Data', fontsize=11, fontweight='bold')
    ax.set_ylabel('NDVI', fontsize=11, fontweight='bold')
    ax.set_title(f"Ciclo {cycle['cycle_num']}: {cycle['start_date'].strftime('%Y-%m-%d')} a {cycle['end_date'].strftime('%Y-%m-%d')}", 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def main():
    """Função principal para demonstração completa."""
    
    print("\n" + "="*80)
    print("TESTE DO MÓDULO PHENOPHASE_LOCAL - Extração Local de Estágios Fenológicos")
    print("="*80 + "\n")
    
    # 1. Cria dados sintéticos
    print("1️⃣  Gerando dados sintéticos de série temporal NDVI...")
    df_ts = create_synthetic_ts_data(num_years=2, cycles_per_year=1, noise_level=0.06)
    print(f"   ✓ {len(df_ts)} observações geradas")
    print(f"   ✓ Período: {df_ts['datetime'].min().strftime('%Y-%m-%d')} a {df_ts['datetime'].max().strftime('%Y-%m-%d')}")
    
    # 2. Extrai sazonalidade
    print("\n2️⃣  Extraindo sazonalidade da série (STD móvel)...")
    seasonality = extract_seasonality_std(df_ts, ndvi_column='NDVI_mean')
    print(f"   ✓ Sazonalidade calculada com janela de 365 dias")
    
    # 3. Identifica ciclos
    print("\n3️⃣  Identificando ciclos de safra (mínimos de NDVI)...")
    cycles = identify_crop_cycles(df_ts, ndvi_column='NDVI_mean', min_cycle_length=60)
    print(f"   ✓ {len(cycles)} ciclo(s) detectado(s)")
    for i, cycle in enumerate(cycles):
        print(f"      Ciclo {i+1}: {cycle['length_days']:.0f} dias "
              f"({cycle['start_date'].strftime('%Y-%m-%d')} a {cycle['end_date'].strftime('%Y-%m-%d')})")
    
    # 4. Ajusta gaussianas e extrai fenometria
    print("\n4️⃣  Ajustando gaussianas em cada ciclo...")
    phenometrics = extract_phenometrics_local(df_ts, ndvi_column='NDVI_mean')
    
    # 5. Imprime resumo
    print("\n5️⃣  Resultados finais:")
    print_phenometrics_summary(phenometrics)
    
    # 6. Plota resultados
    print("6️⃣  Gerando visualizações...")
    
    # Série temporal completa
    fig1, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_ts['datetime'], df_ts['NDVI_mean'], 'o-', color='#003366', linewidth=1.5, markersize=3)
    
    # Marca ciclos detectados
    for cycle in cycles:
        ax.axvline(cycle['start_date'], color='green', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(cycle['end_date'], color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Data', fontsize=11, fontweight='bold')
    ax.set_ylabel('NDVI', fontsize=11, fontweight='bold')
    ax.set_title('Série Temporal Completa de NDVI com Ciclos Detectados', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('teste_serie_temporal.png', dpi=150, bbox_inches='tight')
    print("   ✓ Série temporal salva em: teste_serie_temporal.png")
    
    # Plota cada ciclo individual
    for cycle, fitted_cycle in zip(cycles, phenometrics['cycles']):
        if fitted_cycle.get('fit_success', False):
            fig = plot_cycle_with_gaussian(df_ts, cycle, fitted_cycle,
                                         save_path=f"teste_ciclo_{cycle['cycle_num']}.png")
            print(f"   ✓ Ciclo {cycle['cycle_num']} salvo em: teste_ciclo_{cycle['cycle_num']}.png")
    
    # Plota sazonalidade
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(df_ts['datetime'], df_ts['NDVI_mean'], 'o-', color='#003366', label='NDVI observado', linewidth=1, markersize=2)
    ax1.set_ylabel('NDVI', fontsize=10, fontweight='bold')
    ax1.set_title('Série Temporal Original e STD Móvel', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Converte timestamps para datetime para plotar
    datetimes = pd.to_datetime(seasonality['datetime'])
    ax2.plot(datetimes, seasonality['std_rolling'], '-', color='orange', linewidth=1.5, label='STD Móvel (365 dias)')
    ax2.fill_between(datetimes, seasonality['std_rolling'], color='orange', alpha=0.3)
    ax2.set_xlabel('Data', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Desvio Padrão', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('teste_sazonalidade.png', dpi=150, bbox_inches='tight')
    print("   ✓ Sazonalidade salva em: teste_sazonalidade.png")
    
    print("\n✅ Teste concluído com sucesso!")
    print("="*80 + "\n")
    
    return df_ts, phenometrics


if __name__ == "__main__":
    df_ts, phenometrics = main()
    
    # Mantém os plots abertos
    plt.show()

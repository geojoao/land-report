"""
Módulo local para extração de estágios fenológicos de séries temporais de NDVI.
Substitui a dependência do wcpms do INPE com métodos baseados em gaussiana.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def gaussian(x: np.ndarray, amplitude: float, mean: float, std: float, offset: float) -> np.ndarray:
    """
    Modelo gaussiano para ajuste de dados fenológicos.
    
    Args:
        x: Valores de entrada (dias desde o início)
        amplitude: Amplitude da gaussiana
        mean: Média (centro da gaussiana) - corresponde ao POS (Peak of Season)
        std: Desvio padrão da gaussiana
        offset: Deslocamento vertical (baseline)
    
    Returns:
        Valores da gaussiana
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2)) + offset


def extract_seasonality_std(df_ts: pd.DataFrame, ndvi_column: str = 'NDVI_mean', window: int = 365) -> Dict[str, np.ndarray]:
    """
    Extrai sazonalidade usando desvio padrão móvel.
    
    Args:
        df_ts: DataFrame com série temporal (deve ter coluna 'datetime')
        ndvi_column: Nome da coluna NDVI
        window: Tamanho da janela para cálculo do STD (dias)
    
    Returns:
        Dicionário com informações de sazonalidade
    """
    df_ts = df_ts.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    df_ts = df_ts.sort_values('datetime')
    
    # Calcula STD móvel
    df_ts['std_rolling'] = df_ts[ndvi_column].rolling(window=window, center=True).std()
    
    # Calcula média móvel para remover tendências
    df_ts['mean_rolling'] = df_ts[ndvi_column].rolling(window=window, center=True).mean()
    
    # Normaliza NDVI pela média móvel para remover variações de longo prazo
    df_ts['ndvi_normalized'] = df_ts[ndvi_column] / (df_ts['mean_rolling'] + 1e-6)
    
    return {
        'datetime': df_ts['datetime'].values,
        'ndvi': df_ts[ndvi_column].values,
        'std_rolling': df_ts['std_rolling'].values,
        'mean_rolling': df_ts['mean_rolling'].values,
        'ndvi_normalized': df_ts['ndvi_normalized'].values
    }


def identify_crop_cycles(df_ts: pd.DataFrame, ndvi_column: str = 'NDVI_mean', 
                        min_ndvi_threshold: float = None, prominence: float = 0.15,
                        min_cycle_length: int = 60) -> List[Dict[str, any]]:
    """
    Identifica ciclos de safra usando mínimos locais de NDVI (solo exposto).
    
    Args:
        df_ts: DataFrame com série temporal
        ndvi_column: Nome da coluna NDVI
        min_ndvi_threshold: Limiar de NDVI mínimo para detectar solo exposto (None = auto)
        prominence: Proeminência dos picos para detecção
        min_cycle_length: Comprimento mínimo do ciclo em dias
    
    Returns:
        Lista de dicionários com informações de cada ciclo
    """
    df_ts = df_ts.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    df_ts = df_ts.sort_values('datetime')
    
    ndvi_values = df_ts[ndvi_column].values
    dates = df_ts['datetime'].values
    
    # Suaviza a série para detectar mínimos com mais robustez
    if len(ndvi_values) > 11:
        ndvi_smooth = savgol_filter(ndvi_values, window_length=11, polyorder=2)
    else:
        ndvi_smooth = ndvi_values
    
    # Detecta mínimos locais (inverte a série para usar find_peaks)
    ndvi_inverted = -ndvi_smooth
    min_peaks, min_properties = find_peaks(ndvi_inverted, prominence=prominence)
    
    if len(min_peaks) == 0:
        # Se não encontrar picos com prominence, usa mínimos diretos
        min_peaks = np.argsort(ndvi_smooth)[:max(1, len(ndvi_smooth) // 365)]
        min_peaks = np.sort(min_peaks)
    
    # Se definir threshold, filtra mínimos abaixo dele
    if min_ndvi_threshold is None:
        min_ndvi_threshold = np.percentile(ndvi_values, 20)  # 20º percentil por padrão
    
    min_peaks = [p for p in min_peaks if ndvi_values[p] <= min_ndvi_threshold]
    
    # Agrupa mínimos próximos e extrai o mínimo de cada grupo
    if len(min_peaks) == 0:
        return []
    
    min_peaks = np.array(sorted(set(min_peaks)))
    
    # Agrupa mínimos que estão próximos (menos de min_cycle_length/2 dias)
    grouped_mins = []
    current_group = [min_peaks[0]]
    
    for i in range(1, len(min_peaks)):
        days_diff = (dates[min_peaks[i]] - dates[current_group[-1]]) / np.timedelta64(1, 'D')
        if days_diff < min_cycle_length / 2:
            current_group.append(min_peaks[i])
        else:
            # Seleciona o mínimo do grupo
            min_idx = current_group[np.argmin(ndvi_values[current_group])]
            grouped_mins.append(min_idx)
            current_group = [min_peaks[i]]
    
    # Processa o último grupo
    if current_group:
        min_idx = current_group[np.argmin(ndvi_values[current_group])]
        grouped_mins.append(min_idx)
    
    grouped_mins = np.array(sorted(grouped_mins))
    
    # Cria ciclos entre mínimos consecutivos
    cycles = []
    
    for i in range(len(grouped_mins)):
        if i == 0:
            # Primeiro ciclo começa no primeiro mínimo
            start_idx = grouped_mins[i]
            end_idx = grouped_mins[i + 1] if i + 1 < len(grouped_mins) else len(ndvi_values) - 1
        elif i == len(grouped_mins) - 1:
            # Último ciclo
            start_idx = grouped_mins[i - 1]
            end_idx = grouped_mins[i]
        else:
            # Ciclos do meio
            start_idx = grouped_mins[i]
            end_idx = grouped_mins[i + 1]
        
        cycle_length_days = (dates[end_idx] - dates[start_idx]) / np.timedelta64(1, 'D')
        
        # Filtra ciclos muito curtos
        if cycle_length_days >= min_cycle_length:
            cycles.append({
                'cycle_num': len(cycles) + 1,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'start_date': pd.Timestamp(dates[start_idx]),
                'end_date': pd.Timestamp(dates[end_idx]),
                'length_days': cycle_length_days,
                'start_ndvi': ndvi_values[start_idx],
                'end_ndvi': ndvi_values[end_idx]
            })
    
    return cycles


def fit_gaussian_to_cycle(df_ts: pd.DataFrame, cycle: Dict, ndvi_column: str = 'NDVI_mean') -> Dict[str, any]:
    """
    Ajusta uma gaussiana a um ciclo de safra específico e extrai parâmetros fenológicos.
    
    Args:
        df_ts: DataFrame com série temporal
        cycle: Dicionário de ciclo retornado por identify_crop_cycles
        ndvi_column: Nome da coluna NDVI
    
    Returns:
        Dicionário com parâmetros da gaussiana e estágios fenológicos
    """
    df_ts = df_ts.copy()
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    df_ts = df_ts.sort_values('datetime')
    
    start_idx = cycle['start_idx']
    end_idx = cycle['end_idx']
    
    # Extrai dados do ciclo
    cycle_data = df_ts.iloc[start_idx:end_idx + 1].copy()
    ndvi_cycle = cycle_data[ndvi_column].values
    dates_cycle = cycle_data['datetime'].values
    
    # Converte datas para dias desde o início do ciclo
    days_since_start = np.array([(d - dates_cycle[0]) / np.timedelta64(1, 'D') for d in dates_cycle])
    
    # Parâmetros iniciais para o ajuste
    amplitude_init = np.max(ndvi_cycle) - np.min(ndvi_cycle)
    mean_init = days_since_start[np.argmax(ndvi_cycle)]
    std_init = (days_since_start[-1] - days_since_start[0]) / 4  # Aproximadamente 1/4 do ciclo
    offset_init = np.min(ndvi_cycle)
    
    initial_guess = [amplitude_init, mean_init, std_init, offset_init]
    
    # Define limites para o ajuste
    lower_bounds = [0.01, days_since_start[0], 5, np.min(ndvi_cycle) - 0.1]
    upper_bounds = [1.0, days_since_start[-1], (days_since_start[-1] - days_since_start[0]) / 2, np.max(ndvi_cycle)]
    
    try:
        # Ajusta a gaussiana
        popt, pcov = curve_fit(
            gaussian, 
            days_since_start, 
            ndvi_cycle,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,
            method='trf'
        )
        
        amplitude, mean_pos, std_dev, offset = popt
        
        # Calcula qualidade do ajuste (R²)
        residuals = ndvi_cycle - gaussian(days_since_start, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ndvi_cycle - np.mean(ndvi_cycle)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Converte SOS, POS, EOS para datas reais
        # SOS: ponto onde gaussiana sobe até 25% da amplitude
        # POS: ponto máximo (mean da gaussiana)
        # EOS: ponto onde gaussiana desce até 25% da amplitude
        
        # Calcula o valor de 25% da amplitude acima do offset
        amplitude_25pct = offset + 0.25 * amplitude
        
        # Encontra SOS e EOS resolvendo: f(x) = amplitude_25pct
        # amplitude * exp(-(x-mean)^2 / (2*std^2)) + offset = amplitude_25pct
        # exp(-(x-mean)^2 / (2*std^2)) = 0.25
        # -(x-mean)^2 / (2*std^2) = ln(0.25)
        # (x-mean)^2 = -2 * std^2 * ln(0.25)
        x_offset = np.sqrt(-2 * std_dev ** 2 * np.log(0.25))
        
        sos_days = mean_pos - x_offset
        eos_days = mean_pos + x_offset
        
        # Garante que SOS e EOS estão dentro do ciclo
        sos_days = max(sos_days, days_since_start[0])
        eos_days = min(eos_days, days_since_start[-1])
        
        # Converte para datas reais
        sos_date = pd.Timestamp(dates_cycle[0]) + timedelta(days=float(sos_days))
        pos_date = pd.Timestamp(dates_cycle[0]) + timedelta(days=float(mean_pos))
        eos_date = pd.Timestamp(dates_cycle[0]) + timedelta(days=float(eos_days))
        
        return {
            'cycle_num': cycle['cycle_num'],
            'cycle_start': cycle['start_date'],
            'cycle_end': cycle['end_date'],
            'cycle_length_days': cycle['length_days'],
            'fit_success': True,
            'r_squared': r_squared,
            'gaussian_params': {
                'amplitude': float(amplitude),
                'mean_days': float(mean_pos),  # Dias desde início do ciclo
                'std_dev_days': float(std_dev),
                'offset': float(offset)
            },
            'phenophase_dates': {
                'sos': sos_date,  # Start of Season
                'pos': pos_date,  # Peak of Season
                'eos': eos_date   # End of Season
            },
            'phenophase_values': {
                'sos_ndvi': float(gaussian(sos_days, *popt)),
                'pos_ndvi': float(gaussian(mean_pos, *popt)),
                'eos_ndvi': float(gaussian(eos_days, *popt))
            }
        }
    
    except Exception as e:
        # Se falhar o ajuste, retorna um resultado vazio
        print(f"Erro ao ajustar gaussiana para ciclo {cycle['cycle_num']}: {e}")
        return {
            'cycle_num': cycle['cycle_num'],
            'cycle_start': cycle['start_date'],
            'cycle_end': cycle['end_date'],
            'cycle_length_days': cycle['length_days'],
            'fit_success': False,
            'error': str(e)
        }


def extract_phenometrics_local(df_ts: pd.DataFrame, ndvi_column: str = 'NDVI_mean',
                               min_cycle_length: int = 60,
                               min_ndvi_threshold: Optional[float] = None) -> Dict[str, any]:
    """
    Extrai métricas fenológicas completas de uma série temporal local.
    
    Args:
        df_ts: DataFrame com série temporal (deve ter 'datetime' e coluna NDVI)
        ndvi_column: Nome da coluna NDVI
        min_cycle_length: Comprimento mínimo do ciclo em dias
        min_ndvi_threshold: Limiar para detectar solo exposto (None = auto)
    
    Returns:
        Dicionário com métricas fenológicas e ciclos ajustados
    """
    # Extrai sazonalidade
    seasonality = extract_seasonality_std(df_ts, ndvi_column)
    
    # Identifica ciclos de safra
    cycles = identify_crop_cycles(df_ts, ndvi_column, min_ndvi_threshold, min_cycle_length=min_cycle_length)
    
    if not cycles:
        print("Aviso: Nenhum ciclo de safra detectado na série temporal")
        return {
            'success': False,
            'cycles': [],
            'num_cycles': 0
        }
    
    # Ajusta gaussiana em cada ciclo
    fitted_cycles = []
    for cycle in cycles:
        fitted_cycle = fit_gaussian_to_cycle(df_ts, cycle, ndvi_column)
        fitted_cycles.append(fitted_cycle)
    
    # Calcula estatísticas gerais
    successful_fits = [c for c in fitted_cycles if c.get('fit_success', False)]
    
    if successful_fits:
        mean_r_squared = np.mean([c['r_squared'] for c in successful_fits])
        mean_cycle_length = np.mean([c['cycle_length_days'] for c in successful_fits])
    else:
        mean_r_squared = 0
        mean_cycle_length = 0
    
    return {
        'success': True,
        'num_cycles': len(cycles),
        'num_successful_fits': len(successful_fits),
        'mean_r_squared': float(mean_r_squared),
        'mean_cycle_length_days': float(mean_cycle_length),
        'seasonality': seasonality,
        'cycles': fitted_cycles
    }


def print_phenometrics_summary(phenometrics: Dict) -> None:
    """
    Imprime um resumo dos resultados das métricas fenológicas.
    
    Args:
        phenometrics: Dicionário retornado por extract_phenometrics_local
    """
    print("\n" + "="*80)
    print("RESUMO DE MÉTRICAS FENOLÓGICAS (MÉTODO LOCAL)")
    print("="*80)
    
    if not phenometrics.get('success', False):
        print("Erro: Nenhum ciclo detectado")
        return
    
    print(f"\nCiclos detectados: {phenometrics['num_cycles']}")
    print(f"Ajustes bem-sucedidos: {phenometrics['num_successful_fits']}")
    print(f"R² médio: {phenometrics['mean_r_squared']:.4f}")
    print(f"Comprimento médio do ciclo: {phenometrics['mean_cycle_length_days']:.1f} dias")
    
    print("\n" + "-"*80)
    print("DETALHES DE CADA CICLO:")
    print("-"*80)
    
    for cycle in phenometrics['cycles']:
        if not cycle.get('fit_success', False):
            print(f"\nCiclo {cycle['cycle_num']}: ERRO NO AJUSTE ({cycle.get('error', 'desconhecido')})")
            continue
        
        print(f"\n📊 CICLO {cycle['cycle_num']}:")
        print(f"   Período: {cycle['cycle_start'].strftime('%Y-%m-%d')} a {cycle['cycle_end'].strftime('%Y-%m-%d')}")
        print(f"   Duração: {cycle['cycle_length_days']:.0f} dias")
        print(f"   R² do ajuste: {cycle['r_squared']:.4f}")
        
        print(f"   📈 Parâmetros da Gaussiana:")
        params = cycle['gaussian_params']
        print(f"      - Amplitude: {params['amplitude']:.4f}")
        print(f"      - Desvio Padrão: {params['std_dev_days']:.1f} dias")
        print(f"      - Offset: {params['offset']:.4f}")
        
        phenophase = cycle['phenophase_dates']
        phenophase_val = cycle['phenophase_values']
        print(f"   🌱 Estágios Fenológicos:")
        print(f"      - SOS (Start of Season): {phenophase['sos'].strftime('%Y-%m-%d')} (NDVI={phenophase_val['sos_ndvi']:.4f})")
        print(f"      - POS (Peak of Season):  {phenophase['pos'].strftime('%Y-%m-%d')} (NDVI={phenophase_val['pos_ndvi']:.4f})")
        print(f"      - EOS (End of Season):   {phenophase['eos'].strftime('%Y-%m-%d')} (NDVI={phenophase_val['eos_ndvi']:.4f})")
    
    print("\n" + "="*80 + "\n")

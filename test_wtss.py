import os
import sys
import math
import geopandas as gpd
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import contextily as ctx

import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sqlalchemy import create_engine

from pystac_client import Client
import pystac_client
import shapely

import wtss
from shapely.geometry import Polygon, Point
import random
from scipy.signal import savgol_filter

from skimage.transform import resize
from phenophase import extract_phenometrics, print_phenometrics_summary, plot_diagnostic

from matplotlib.backends.backend_pdf import PdfPages
import kaleido

import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

import requests, zipfile, io

from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

from collections import defaultdict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def print_status(message):
    """Prints a message to stderr to show in the console but not in the report."""
    print(message, file=sys.stderr, flush=True)

def get_and_plot_areal_ts_wtss(gleba_4326, gleba_id, planting_period=None, harvesting_period=None):
    """
    Busca e plota a série temporal média de 6 anos para a gleba inteira
    usando a função summarize (geom=) do WTSS e adiciona SOS, POS e EOS.
    
    Args:
        gleba_4326 (shapely.geometry): A geometria da gleba em EPSG:4326.
        gleba_id (str): O ID da gleba para o título.
        planting_period (tuple, optional): Tupla com (start_date, end_date) do plantio.
        harvesting_period (tuple, optional): Tupla com (start_date, end_date) da colheita.
    """
    # Inicializar o cubo WTSS
    bdc_wtss_link = 'https://data.inpe.br/bdc/wtss/v4/'
    colecao_modis = 'mod13q1-6.1'  # Ajuste se necessário
    servico_wtss = wtss.WTSS(bdc_wtss_link)
    cubo = servico_wtss[colecao_modis]

    print_status("   - Definindo período de 6 anos para série temporal média (WTSS)...")
    end_date_dt = datetime.now()
    start_date_dt = end_date_dt - relativedelta(years=6)
    start_date_str = start_date_dt.strftime('%Y-%m-%d')

    start_date_pheno_dt = end_date_dt - relativedelta(years=1)
    start_date_pheno_str = start_date_pheno_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    
    print_status(f"   - Buscando série temporal sumarizada de {start_date_str} a {end_date_str}...")

    # Simplifica a geometria para evitar URLs muito longas que causam erro 400 (Bad Request)
    gleba_geojson = gleba_4326.__geo_interface__
    timeseries = cubo.ts(
        attributes=('NDVI',), # Fetch SCL for cloud masking
        geom=gleba_geojson,
        start_date=start_date_str,
        end_date=end_date_str,
    )
    
    # 2. Chamar summarize
    print_status("   - Sumarizando dados...")
    summarize = timeseries.summarize()
    
    # 3. Plotar (usando o estilo do exemplo fornecido)
    print_status("   - Plotando gráfico da média...")
    
    # Define uma paleta de cores mais corporativa
    corporate_blue = '#003366'
    fill_color = '#e6f0f7'
    grid_color = '#cccccc'
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Extrai o DataFrame com todos os pixels usando o método .df()
    df = summarize.df()

    # --- This is the conversion snippet ---
    df['band_agg'] = df['attribute'] + '_' + df['aggregation']
    wide_df = df.pivot(index='datetime', columns='band_agg', values='value')
    wide_df = wide_df.reset_index()
    wide_df.columns.name = None
    df_ts = wide_df

    # --- Smoothing Step ---
    for col in df_ts.columns:
        if col != 'datetime' and pd.api.types.is_numeric_dtype(df_ts[col]):
            if len(df_ts[col]) > 5:
                df_ts[col] = savgol_filter(df_ts[col], window_length=5, polyorder=2)

    # Verifica se há dados para plotar
    if not df_ts.empty:
        x_data = df_ts.datetime
        y_data = df_ts['NDVI_mean']
        ax.plot(x_data, y_data, color=corporate_blue, linewidth=2.0)
        ax.fill_between(x_data, y_data, color=fill_color, alpha=0.7)

    # --- Adiciona as janelas de plantio e colheita ---
    if planting_period:
        try:
            plant_start, plant_end = pd.to_datetime(planting_period[0]), pd.to_datetime(planting_period[1])
            ax.axvspan(plant_start, plant_end, color='green', alpha=0.2, label='Janela de Plantio Esperada')
        except (TypeError, IndexError, ValueError) as e:
            print_status(f"   - AVISO: Formato inválido para planting_period. Esperado (start, end). Erro: {e}")

    if harvesting_period:
        try:
            harvest_start, harvest_end = pd.to_datetime(harvesting_period[0]), pd.to_datetime(harvesting_period[1])
            ax.axvspan(harvest_start, harvest_end, color='orange', alpha=0.2, label='Janela de Colheita Esperada')
        except (TypeError, IndexError, ValueError) as e:
            print_status(f"   - AVISO: Formato inválido para harvesting_period. Esperado (start, end). Erro: {e}")

    # --- Adiciona SOS, POS e EOS ---
    print_status("   - Calculando métricas fenológicas (SOS, POS, EOS) com método V2...")
    
    # Usa o novo método v2 para extrair fenometria
    # Otimizado para detectar múltiplas safras (safra e safrinha)
    phenometrics = extract_phenometrics(
        df_ts, 
        ndvi_column='NDVI_mean',
        min_cycle_length_days=45,  # Reduzido para capturar safrinhas mais curtas
        quality_threshold=0.55,     # Mais permissivo
        smoothing_method='both',    # Combina median + savgol
        quantile_trough=25          # Mais sensível
    )
    
    # Se o método v2 funcionou, plota os estágios fenológicos
    if phenometrics.get('success', False) and phenometrics['cycles']:
        # Plota TODOS os ciclos bem-sucedidos
        successful_cycles = [c for c in phenometrics['cycles'] if c.get('fit_success', False)]
        
        if successful_cycles:
            colors_cycle = ['blue', 'red', 'purple', 'green', 'orange', 'brown']
            
            for idx, cycle in enumerate(successful_cycles):
                color_base = colors_cycle[idx % len(colors_cycle)]
                phenophase = cycle['phenophase_dates']
                
                # Adiciona linhas verticais para SOS, POS e EOS
                ax.axvline(phenophase['sos'], color=color_base, linestyle='--', alpha=0.6, linewidth=1.5)
                ax.axvline(phenophase['pos'], color=color_base, linestyle='-', alpha=0.7, linewidth=2)
                ax.axvline(phenophase['eos'], color=color_base, linestyle='--', alpha=0.6, linewidth=1.5)
                
                # Anotações para primeiros ciclos
                if idx < 2:
                    ax.annotate(f'C{idx+1}', 
                               xy=(phenophase['pos'], ax.get_ylim()[1] * 0.85), 
                               xytext=(0, -5), textcoords='offset points', 
                               ha='center', va='top', fontsize=8, color=color_base, fontweight='bold')
            
            print_status(f"   - Ciclos detectados: {phenometrics['num_cycles_detected']}")
            print_status(f"   - Ajustes bem-sucedidos: {phenometrics['num_successful_fits']}")
            print_status(f"   - R² médio: {phenometrics['mean_r_squared']:.4f}")
    else:
        print_status("   - ⚠️ Nenhum ciclo bem-sucedido detectado com método V2")

    # Ajusta a legenda para evitar duplicatas
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Ajustes de estilo
    ax.set_title(f'Série Temporal Média de NDVI (Últimos 6 anos) - Gleba {gleba_id}', loc='left', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data', fontsize=10)
    ax.set_ylabel('NDVI Médio', fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='-', alpha=0.5, color=grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# Código para testar com WKT
if __name__ == "__main__":

    # Solicitar WKT do usuário
    wkt_input = "POLYGON((-47.0 -22.0, -47.0 -21.0, -46.0 -21.0, -46.0 -22.0, -47.0 -22.0))"
    
    # Converter WKT para shapely geometry
    from shapely import wkt
    gleba_4326 = wkt.loads(wkt_input)
    
    # ID da gleba
    gleba_id = 'teste_123'
    
    # Períodos opcionais
    planting_period = None  # Pode adicionar input se quiser
    harvesting_period = None
    
    # Chamar a função
    fig = get_and_plot_areal_ts_wtss(gleba_4326, gleba_id, planting_period, harvesting_period)
    
    # Mostrar o plot
    plt.show()

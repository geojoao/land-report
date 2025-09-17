# %% [markdown]
# # <span style="text-align:center; color:#336699">Visualização integrada de dados</span>
# <hr style="border:2px solid #0077b9;">
# 
# 
# <br/>
# <div style="text-align: center;font-size: 90%;">
#     Thales Sehn Körting, Marcos Adami, Gilberto Queiroz, Karine Ferreira 
#     <br/><br/>
#     Divisão de Observação da Terra e Geoinformática, Instituto Nacional de Pesquisas Espaciais (INPE)
#     <br/>
#     Avenida dos Astronautas, 1758, Jardim da Granja, São José dos Campos, SP 12227-010, Brazil
#     <br/><br/>
#     Contato: <a href="https://geo-credito-rural.github.io/">Equipe - Geo Credito Rural</a>
#     <br/><br/>
#     Última atualização: 29 de Setembro de 2024
# </div>
# 
# <br/>
# 
# <div style="text-align: justify;  margin-left: 25%; margin-right: 25%;">
# <b>Resumo:</b> Este Jupyter Notebook é parte do material da capacitação "Monitoramento de Operações de Crédito Rural por meio de Geotecnologias". Os trechos de código apresentados são um protótipo da integração dos módulos anteriores. O script parte de um polígono de uma gleba, obtendo informações sobre a mesma, e realiza o cruzamento com dados de malha fundiária, mapas de uso e cobertura da terra, observação de séries temporais pontuais e por subregiões homogênes, e finaliza com a apresentação de estatísticas dos recortes das imagens ao longo do tempo.        
# </div>    
# 
# <br/>

# %% [markdown]
# ## Importação das bibliotecas
# ---

# %%
import os
import sys
import math
import geopandas as gpd
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# Use a non-interactive backend to prevent plots from being displayed.
matplotlib.use('Agg')
import contextily as ctx

import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from datetime import datetime

# conexão e consultas SQL com banco de dados no geolab
from sqlalchemy import create_engine
#import sicor

# visualização de cenas usando TMS
from pystac_client import Client
import pystac_client
import shapely

# séries temporais
import wtss
from shapely.geometry import Polygon
from shapely.geometry import Point
import random
from wcpms import *

# processamento de imagens
from skimage.transform import resize

# exportação do relatório em PDF
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import kaleido

import xml.etree.ElementTree as ET
import requests, zipfile, io

# desabilitar warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# %%


def recursive_extract(elem, ns, prefix=''):
    """Recursively extract text from XML elements into a flat dict."""
    data = {}
    for child in elem:
        tag = child.tag.split('}', 1)[-1]  # Remove namespace
        key = f"{prefix}_{tag}" if prefix else tag
        if len(child) == 0:
            # Leaf node
            text = child.text.strip() if child.text else None
            data[key] = text
        else:
            # Recursive call
            data.update(recursive_extract(child, ns, key))
    return data

def parse_xml_to_gdf_all_fields(xml_string):
    ns = {'ns': 'http://www.bcb.gov.br/MES/COR0001.xsd'}
    root = ET.fromstring(xml_string)
    
    cor = root.find('.//ns:COR0001', ns)
    if cor is None:
        raise ValueError("No COR0001 element found in XML")
    
    # Extract all fields recursively
    data = recursive_extract(cor, ns)
    
    # Extract polygon points from Grupo_COR0001_Gleba inside Grupo_COR0001_DestcFincmnt
    polygon_points = []
    destc = cor.find('ns:Grupo_COR0001_DestcFincmnt', ns)
    if destc is not None:
        gleba = destc.find('ns:Grupo_COR0001_Gleba', ns)
        if gleba is not None:
            for pt in gleba.findall('ns:Grupo_COR0001_PontoPolg', ns):
                lat = pt.find('ns:LatPonto', ns)
                lon = pt.find('ns:LongPonto', ns)
                if lat is not None and lon is not None:
                    try:
                        lat_f = float(lat.text.strip())
                        lon_f = float(lon.text.strip())
                        polygon_points.append((lon_f, lat_f))
                    except Exception:
                        pass
    
    if not polygon_points:
        raise ValueError("No polygon points found in XML")
    
    polygon = Polygon(polygon_points)
    
    # Create GeoDataFrame with one row
    gdf = gpd.GeoDataFrame([data], geometry=[polygon], crs="EPSG:4326")
    
    return gdf


try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:
    ScaleBar = None  # Optional if scalebar not installed



# ---------------------------------------------------
# Function to download and load IBGE municipalities
# ---------------------------------------------------
def load_ibge_municipalities():
    url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/Brasil/BR_Municipios_2024.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    shp_path = [f for f in z.namelist() if f.endswith(".shp")][0]
    z.extractall("ibge_municipios")  # Extract locally
    gdf = gpd.read_file(f"ibge_municipios/{shp_path}")
    gdf = gdf.to_crs("EPSG:4326")  # ensure WGS84
    return gdf


# ---------------------------------------------------
# Function to plot rural property with minimap + info
# ---------------------------------------------------
def plot_rural_property(property_gdf, municipalities_gdf=None):
    """
    Generate a rural property geographical location map.
    
    Parameters
    ----------
    property_gdf : GeoDataFrame
        GeoDataFrame containing rural property geometry.
        Columns 'name', 'city', 'state' are expected.
    municipalities_gdf : GeoDataFrame, optional
        GeoDataFrame of municipalities for minimap.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
    """

    # Compute centroid
    centroid = property_gdf.geometry.centroid.iloc[0]
    lon, lat = centroid.x, centroid.y

    # Create figure layout
    fig = plt.figure(figsize=(12, 8))

    # Main map
    ax_main = fig.add_axes([0.05, 0.05, 0.65, 0.9])
    property_gdf.to_crs(epsg=3857).plot(ax=ax_main, facecolor="none", edgecolor="red", linewidth=2)
    ctx.add_basemap(ax_main, source=ctx.providers.Esri.WorldImagery)
    ax_main.set_axis_off()

    # Add scalebar
    if ScaleBar is not None:
        scalebar = ScaleBar(1, location='lower right')
        ax_main.add_artist(scalebar)

    # North arrow
    ax_main.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.05),
                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
                     ha='center', va='center', fontsize=12,
                     xycoords=ax_main.transAxes)

    # Minimap
    if municipalities_gdf is not None:
        ax_mini = fig.add_axes([0.72, 0.65, 0.25, 0.25])
        municipalities_gdf.to_crs(epsg=3857).plot(ax=ax_mini, facecolor="none", edgecolor="black")
        property_gdf.to_crs(epsg=3857).plot(ax=ax_mini, facecolor="red", alpha=0.5)
        ax_mini.set_axis_off()
        ax_mini.set_title("Municipality", fontsize=10)

    # Info panel
    ax_info = fig.add_axes([0.72, 0.05, 0.25, 0.55])
    ax_info.axis("off")
    info_text = f"""
General information about the rural property

City: {property_gdf['city'].iloc[0]}
State: {property_gdf['state'].iloc[0]}
Centroid: {lat:.4f}, {lon:.4f}
"""
    ax_info.text(0, 1, info_text, va="top", fontsize=10)

    return fig

try:
    # Uncomment to load land from xml:
    with open('locks.xml', 'r', encoding='utf-8') as f: #encoding='utf-16be') as f:
        xml_content = f.read()
    print("1. Parsing XML and creating GeoDataFrame...")
    gdf = parse_xml_to_gdf_all_fields(xml_content)
except FileNotFoundError:
    print("ERRO: O arquivo 'locks.xml' não foi encontrado. Certifique-se de que ele está no diretório correto ou montado no container Docker.")
    sys.exit(1) # Exit with an error code

# Uncomment to load land from kml:
# gdf = gpd.read_file('glebas/sto_angelo.kml', driver='KML')

# Load IBGE municipalities
municipios = load_ibge_municipalities()

# Spatial join with IBGE municipalities
joined = gpd.sjoin(gdf, municipios, how="left", predicate="intersects")

# IBGE schema: "NM_MUN" (municipality), "NM_UF" (state)
joined["city"] = joined["NM_MUN"]
joined["state"] = joined["NM_UF"]

# Plot
fig_localization = plot_rural_property(joined, municipalities_gdf=municipios[municipios["NM_MUN"] == joined["city"].iloc[0]])


# %%
# indicar um valor específico de REF_BACEN
ref_bacen = 513438999
# sugestões de gleba com cruzamentos válidos no banco de dados
# 514003202, 515100357, 514043455, 518712673, 514695166, 513971982, 513971129, 518418726, 514021307, 514064625, 518084698, 516203668, 513438999

# definir intervalo de datas para análise
data_inicial = '2022-01-01'
data_final = '2022-12-31'


# %%
# definição de constantes para executar o script
# definir o total de pontos aleatórios por gleba para 
# visualização das séries temporais
total_pontos = 15
largura_figuras = 18

bdc_stac_link = 'https://data.inpe.br/bdc/stac/v1'
bdc_wtss_link = 'https://data.inpe.br/bdc/wtss/v4/'
bdc_wcpms_url = 'https://data.inpe.br/bdc/wcpms'
colecao_s2 = 'S2-16D-2'
colecao_cbers = 'CBERS4-WFI-16D-2'

# %% [markdown]
# ## Carga da base de dados, contendo glebas, mapas de referência de uso e cobertura do solo
# ---

# %%
# Definir caminho raiz
# root_path = '/kaggle/input/basededados'
root_path = './'

# estratégia 3
gleba = gdf.head(1)

# realizar a conversão de sistema de referencia espacial
# armazenando em outra variável
crs_100000 = '+proj=aea +lon_0=-58.1835937 +lat_1=-36.5394082 +lat_2=-3.0767001 +lat_0=-1 9.8080541 +datum=WGS84 +units=m +no_defs'
gleba_4326 = gleba.to_crs('EPSG:4326').iloc[0]
gleba_100000 = gleba.to_crs(crs_100000).iloc[0]

# para cruzar com o mapbiomas
# Carregar o CSV de classes do MapBiomas a partir de uma string para evitar download/unzip
mapbiomas_legend_csv = """Class_ID	Level	Description	Descricao	Color
1	1	Forest	Floresta	#32a65e
3	2	Forest Formation	Formação Florestal	#1f8d49
4	2	Savanna Formation	Formação Savânica	#7dc975
5	2	Mangrove	Mangue	#04381d
6	2	Floodable Forest	Floresta Alagável	#026975
49	2	Wooded Sandbank Vegetation	Restinga Arbórea	#02d659
10	1	Herbaceous and Shrubby Vegetation	Vegetação Herbácea e Arbustiva	#ad975a
11	2	Wetland	Campo Alagado e Área Pantanosa	#519799
12	2	Grassland	Formação Campestre	#d6bc74
32	2	 Hypersaline Tidal Flat	Apicum	#fc8114
29	2	 Rocky Outcrop	Afloramento Rochoso	#ffaa5f
50	2	 Herbaceous Sandbank Vegetation	Restinga Herbácea	#ad5100
14	1	 Farming	Agropecuária	#FFFFB2
15	2	 Pasture	Pastagem	#edde8e
18	2	 Agriculture	Agricultura	#E974ED
19	3	 Temporary Crop	Lavoura Temporária	#C27BA0
39	4	 Soybean	Soja	#f5b3c8
20	4	 Sugar cane	Cana	#db7093
40	4	 Rice	Arroz	#c71585
62	4	 Cotton (beta)	Algodão (beta)	#ff69b4
41	4	 Other Temporary Crops	Outras Lavouras Temporárias	#f54ca9
36	3	 Perennial Crop	Lavoura Perene	#d082de
46	4	 Coffee	Café	#d68fe2
47	4	 Citrus	Citrus	#9932cc
35	4	 Palm Oil	Dendê	#9065d0
48	4	 Other Perennial Crops	Outras Lavouras Perenes	#e6ccff
9	2	 Forest Plantation	Silvicultura	#7a5900
21	2	 Mosaic of Uses	Mosaico de Usos	#ffefc3
22	1	 Non vegetated area	Área não Vegetada	#d4271e
23	2	 Beach, Dune and Sand Spot	Praia, Duna e Areal	#ffa07a
24	2	 Urban Area	Área Urbanizada	#d4271e
30	2	 Mining	Mineração	#9c0027
75	2	Photovoltaic Power Plant (beta)	Usina Fotovoltaica (beta)	#c12100
25	2	 Other non Vegetated Areas	Outras Áreas não Vegetadas	#db4d4f
26	1	 Water	Corpo D'água	#0000FF
33	2	 River, Lake and Ocean	Rio, Lago e Oceano	#2532e4
31	2	 Aquaculture	Aquicultura	#091077
27	1	 Not Observed	Não observado	#ffffff
"""
classes_mapbiomas = pd.read_csv(io.StringIO(mapbiomas_legend_csv), sep='\t')
print("2. Loaded local data sources (shapefiles, rasters).")

# Carregar o raster
raster_mapbiomas = "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_2024.tif"

# %% [markdown]
# ## Visualização da gleba escolhida e de métricas espaciais
# ---

# %%
gleba_100000.geometry

# %%
# alterar o srid em variável auxiliar, para o cálculo das métricas
print("3. Calculating spatial metrics for the property...")
print('Atributos da gleba')
print('- Area (ha):', gleba_100000.geometry.area/10000)
print('- Perímetro:', gleba_100000.geometry.length)
print('- Razão perímetro/área:', gleba_100000.geometry.length / gleba_100000.geometry.area)
circle = gleba.minimum_bounding_circle()
print('- Relação com círculo:', gleba_100000.geometry.area / circle.to_crs(crs_100000).iloc[0].area)
print('- Relação com quadrado:', gleba_100000.geometry.area / gleba_100000.geometry.minimum_rotated_rectangle.area)

# %%
print(type(gleba_4326))
print(gleba_4326.geometry.is_valid)
print(gleba_4326.geometry.is_empty)

gleba_4326 = gleba_4326.geometry.buffer(0)

# %%
import rioxarray
import shapely.geometry
import numpy as np
import pyproj
from shapely.ops import transform

def reproject_geom(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)

def clip_cog_raster_by_polygon(polygon, raster_cog_path):
    """
    Clip a COG raster by polygon using xarray.sel to subset by bounding box first.

    Parameters:
    - polygon: shapely.geometry.Polygon in EPSG:4326
    - raster_cog_path: str, path or URL to COG raster

    Returns:
    - clipped raster (rioxarray.DataArray)
    - bounding box tuple (minx, miny, maxx, maxy) in EPSG:4326
    """
    # Open raster
    raster = rioxarray.open_rasterio(raster_cog_path, masked=True).squeeze(dim='band')
    raster_crs = raster.rio.crs
    
    # Validate polygon
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError("Input polygon is empty after validation.")
    
    # Get polygon bounding box in EPSG:4326
    minx, miny, maxx, maxy = polygon.bounds
    
    # Subset raster by bounding box using .sel()
    # This assumes raster coords are named 'x' and 'y' and in EPSG:4326
    # If raster CRS is not EPSG:4326, we need to reproject raster or polygon accordingly
    if raster_crs.to_string() != 'EPSG:4326':
        # Reproject polygon to raster CRS for clipping
        polygon_proj = reproject_geom(polygon, "EPSG:4326", raster_crs)
        
        # Reproject bounding box corners to raster CRS for .sel()
        project = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True).transform
        minx_proj, miny_proj = project(minx, miny)
        maxx_proj, maxy_proj = project(maxx, maxy)
        
        # Subset raster by bounding box in raster CRS
        raster_subset = raster.sel(
            x=slice(minx_proj, maxx_proj),
            y=slice(maxy_proj, miny_proj)  # y usually descending
        )
    else:
        polygon_proj = polygon
        raster_subset = raster.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny)  # y descending
        )
    
    # Clip raster subset by polygon
    clipped = raster_subset.rio.clip([shapely.geometry.mapping(polygon_proj)], raster_crs, from_disk=True, all_touched=True)
    
    # Get bounds of clipped raster in raster CRS
    bounds_proj = clipped.rio.bounds()
    
    # Reproject bounds to EPSG:4326 if needed
    if raster_crs.to_string() != 'EPSG:4326':
        project_back = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True).transform
        minx_b, miny_b = project_back(bounds_proj[0], bounds_proj[1])
        maxx_b, maxy_b = project_back(bounds_proj[2], bounds_proj[3])
        bbox_4326 = (minx_b, miny_b, maxx_b, maxy_b)
    else:
        bbox_4326 = bounds_proj
    
    return clipped, bbox_4326

# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot(imagem_cortada, bbox, classes_df, largura_figuras=10):
    """
    Plot the clipped xarray raster with colors and legend from the dataframe.

    Parameters:
    - imagem_cortada: xarray.DataArray with raster data (values correspond to Class_ID)
    - bbox: tuple or list with bounding box coordinates (xmin, ymin, xmax, ymax)
    - classes_df: pandas.DataFrame containing legend info. Can be either:
        - classes_malha_fundiaria with columns ['Class_ID', 'Label', 'Descricao', 'Color']
        - classes_mapbiomas with columns ['Class_ID', 'Level', 'Description', 'Descricao', 'Color']
    - largura_figuras: width of the figure in inches (default 10)

    Returns:
    - fig: matplotlib Figure object of the plot
    - ax: matplotlib Axes object
    """
    fig, ax = plt.subplots(figsize=(largura_figuras, 4))
    
    data = imagem_cortada.values

    # Determine which columns to use for label and color
    if {'Class_ID', 'Label', 'Descricao', 'Color'}.issubset(classes_df.columns):
        # malha_fundiaria style
        class_ids = classes_df['Class_ID'].values
        colors = classes_df['Color'].values
        labels = classes_df['Label'].values
    elif {'Class_ID', 'Level', 'Description', 'Descricao', 'Color'}.issubset(classes_df.columns):
        # mapbiomas style
        class_ids = classes_df['Class_ID'].values
        colors = classes_df['Color'].values
        labels = classes_df['Description'].values
    else:
        raise ValueError("DataFrame columns do not match expected formats.")

    color_dict = dict(zip(class_ids, colors))

    rgb_image = np.empty(data.shape + (4,), dtype=np.float32)
    rgb_image[:] = [1, 1, 1, 0]  # transparent background

    for class_id, hex_color in color_dict.items():
        mask = data == class_id
        if np.any(mask):
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
            rgb_image[mask] = [r, g, b, 1]

    xmin, ymin, xmax, ymax = bbox
    ax.imshow(rgb_image, extent=[xmin, xmax, ymin, ymax], origin='upper')

    patches = []
    for label, color in zip(labels, colors):
        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)

    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Raster Plot with Legend')

    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    return fig, ax

# %%
classes_mapbiomas

# %% [markdown]
# ## Cruzamento da gleba com MapBiomas
# 
# %%
print("5. Clipping MapBiomas land cover raster...")
imagem_cortada, bbox = clip_cog_raster_by_polygon(gleba_4326, raster_mapbiomas)

# plotar o mapa e mostrar a tabela
plt.figure(figsize=(largura_figuras, 4))
fig_mapbiomas, _ = plot(imagem_cortada, bbox, classes_mapbiomas)
fig_mapbiomas.set_size_inches(largura_figuras, 8)
# %% [markdown]
# ## Geração e visualização de pontos aleatórios dentro da gleba
# ---

# %%
import random
from shapely.geometry import Point, Polygon, MultiPolygon

def generate_grid_points(geometry, approx_num_points):
    """
    Generate a grid of points inside a polygon or multipolygon geometry.

    Parameters:
    - geometry: shapely Polygon or MultiPolygon, or GeoSeries with one polygon
    - approx_num_points: int, approximate number of points to generate

    Returns:
    - points: list of shapely.geometry.Point objects inside the geometry
    """
    minx, miny, maxx, maxy = geometry.bounds
    
    # Estimate grid spacing to get approximately the desired number of points
    # This is a rough estimation based on the bounding box area
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area == 0: return []
    
    spacing = np.sqrt(bbox_area / approx_num_points)
    
    # Create grid coordinates
    x_coords = np.arange(minx + spacing/2, maxx, spacing)
    y_coords = np.arange(miny + spacing/2, maxy, spacing)

    points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if geometry.contains(point):
                points.append(point)

    return points

print(f"6. Generating a grid of approximately {total_pontos} points within the property...")
pontos_aleatorios = generate_grid_points(gleba_4326, total_pontos)
print(f"   - Generated {len(pontos_aleatorios)} points.")

# Update total_pontos to the actual number of generated points
total_pontos = len(pontos_aleatorios)

list_x = [float(point.x) for point in pontos_aleatorios]
list_y = [float(point.y) for point in pontos_aleatorios]

plt.figure(figsize=(largura_figuras, 8))
# gpd.GeoSeries(gleba.geometry).plot(color='red', alpha=0.25)
gpd.GeoSeries(gleba_4326).plot(color='red', alpha=0.25)
plt.scatter(list_x, list_y)
for i, p in enumerate(pontos_aleatorios):
    plt.text(p.x, p.y, str(i + 1), fontsize=9, ha='right')
plt.title(f'{total_pontos} pontos distribuídos na gleba')
fig_pontos_aleatorios = plt.gcf()


# %% [markdown]
# ## Visualização de séries temporais com o WTSS (*Web Time Series Service*)
# ---

# %%
# carregar o serviço de séries temporais e o cubo Sentinel-2
servico_wtss = wtss.WTSS(bdc_wtss_link)
cubo_s2 = servico_wtss[colecao_s2]
cubo_s2

# %%
# carregar o cubo de dados alternativo, 64m de resolução espacial, CBERS WFI
cubo_wfi = servico_wtss[colecao_cbers]
cubo_wfi

# %%
vetor_ts_s2 = []
vetor_ts_wfi = []
print(f"7. Fetching time series data for {total_pontos} points...")
for ponto in pontos_aleatorios:
    ts_s2 = cubo_s2.ts(attributes=('NDVI'),
                       latitude=float(ponto.y), 
                       longitude=float(ponto.x),
                       start_date=data_inicial,
                       end_date=data_final)
    vetor_ts_s2.append(ts_s2)

    ts_wfi = cubo_wfi.ts(attributes=('NDVI'),
                         latitude=float(ponto.y), 
                         longitude=float(ponto.x),
                         start_date=data_inicial,
                         end_date=data_final)
    vetor_ts_wfi.append(ts_wfi)

# %%
fig_time_series = plt.figure(figsize=(largura_figuras, 4))
for i, (s2, wfi) in enumerate(zip(vetor_ts_s2, vetor_ts_wfi)):
    plt.plot(s2.timeline, np.array(s2.NDVI)/10000, color='tab:green', alpha=0.7, linewidth=1, label=f'Point {i+1} (Sentinel)' if i == 0 else "")
    plt.plot(wfi.timeline,np.array(wfi.NDVI)/10000, color='tab:red', alpha=0.7, linewidth=1, label=f'Point {i+1} (CBERS)' if i == 0 else "")
plt.grid()
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=45, fontsize=10)
plt.ylim([0.01, 1])
plt.title(f'{total_pontos} séries temporais NDVI para gleba REF_BACEN {ref_bacen}, entre {data_inicial} e {data_final}')
plt.legend();

# %% [markdown]
# ## Visualização de todas as séries temporais pertencentes à gleba
# ---

# %% [markdown]
# ## Visualização das métricas fenológicas

# %%
get_description(url=bdc_wcpms_url)

def retry_request(func, max_retries=3, delay=10, *args, **kwargs):
    """
    A wrapper to retry a function that makes a network request.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print(f"   - Attempt {attempt + 1} of {max_retries} failed: {e}. Retrying in {delay} seconds...")
            if attempt + 1 == max_retries:
                print("   - Max retries reached. Skipping this point.")
                return None
            time.sleep(delay)
    return None


# %%
cubo_fenologia = cube_query(collection=colecao_s2,
                            start_date=data_inicial,
                            end_date=data_final,
                            freq='16D',
                            band='NDVI')

vetor_phenometrics = []
print(f"8. Fetching phenological metrics for {total_pontos} points...")
for i, ponto in enumerate(pontos_aleatorios):
    print(f" - Processing point {i+1}/{total_pontos}...")
    phenometrics_ponto = retry_request(get_phenometrics,
                                       url=bdc_wcpms_url,
                                       cube=cubo_fenologia,
                                       latitude=ponto.y,
                                       longitude=ponto.x)
    if phenometrics_ponto:
        vetor_phenometrics.append(phenometrics_ponto)

# %%
# visualizar as séries temporais individualmente
# junto com as métricas fenológicas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_phenometrics(cube, ds_phenos):
    """
    Plot vegetation index time series with phenological stages.

    Parameters:
    - cube: dict with keys like 'collection', 'band', 'start_date', 'end_date', 'freq'
    - ds_phenos: dict with keys 'timeseries' and 'phenometrics' as described

    Returns:
    - None (plots on current matplotlib figure)
    """
    # Parse timeline dates
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in ds_phenos['timeseries']['timeline']]
    values = ds_phenos['timeseries']['values']

    phenos = ds_phenos['phenometrics']

    # Plot time series
    plt.plot(dates, values, marker='o', label=f"{cube['band']} time series")

    # Define phenological stages to mark: keys with '_t' are dates, '_v' are values
    # We'll mark stages with vertical lines at dates and annotate with stage name and value
    stages = {
        'sos': 'Start of Season',
        'pos': 'Peak of Season',
        'eos': 'End of Season',
        'vos': 'Vegetation Off Season',
        # Add more if needed
    }

    for key_prefix, label in stages.items():
        date_key = f"{key_prefix}_t"
        value_key = f"{key_prefix}_v"
        if date_key in phenos and phenos[date_key] is not None:
            # Convert date string to datetime
            dt = datetime.strptime(phenos[date_key][:10], '%Y-%m-%d')
            val = phenos.get(value_key, None)
            plt.axvline(dt, color='red', linestyle='--', alpha=0.7)
            if val is not None:
                plt.text(dt, val, f"{label}\n{val:.0f}", rotation=90, verticalalignment='bottom', color='red', fontsize=8)

    plt.xlabel('Date')
    plt.ylabel(cube['band'])
    plt.title(f"{cube['collection']} {cube['band']} from {cube['start_date']} to {cube['end_date']}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()


vetor_figs_phenometrics = []

for i, ponto in enumerate(vetor_phenometrics):
    fig_phenometrics = plt.figure(figsize=(largura_figuras, 5))
    plot_phenometrics(cube=cubo_fenologia, 
                      ds_phenos=ponto)
    plt.title(f'Métricas fenológicas do ponto {i+1}/{total_pontos} para gleba REF_BACEN {ref_bacen}, entre {data_inicial} e {data_final}')
    vetor_figs_phenometrics.append(fig_phenometrics)

# %% [markdown]
# ## Visualização dos recortes e estatísticas intermediárias
# ---
# 
# Cada imagem é um recorte disponível na série temporal, acompanhado de um gráfico com as seguintes medidas:
# * **nv**: pixels identificados como nuvens
# * **vg**: pixels identificados com alto índice de vegetação, pelo NDVI *normalized difference vegetation index*
# * **qm**: pixels identificados com alto índice de queimadas, via NBR *normalized burn ratio*
# * **tx**: índice de textura relacionado com desvio padrão dos pixels
# * **um**: pixels identificados com alto índice de água, pelo NDWI *normalized difference water index*
# * **so**: pixels identificados com alto índice de solo exposto, via SCI *soil composition index*
# 
# A composição colorida é padrão, utilizando o contraste de raiz quadrada, com base nos máximos e mínimos locais, da seguinte forma:
# * Canal vermelho visualiza a banda **NIR (B08)**
# * Canal verde visualiza a banda **SWIR (B11)**
# * Canal azul visualiza a banda **Red (B04)**

# %%
import rasterio
from rasterio.mask import mask
import numpy as np
import time
from rasterio.errors import RasterioIOError
from rasterio._err import CPLE_AppDefinedError

def retry_rasterio_open(href, max_retries=5, delay=15):
    """
    A wrapper to retry rasterio.open for network-related errors.
    """
    for attempt in range(max_retries):
        try:
            # Set a timeout for the connection using GDAL config options
            with rasterio.Env(GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_CONNECTTIMEOUT=20):
                return rasterio.open(href)
        except (RasterioIOError, CPLE_AppDefinedError) as e:
            # Check for common network error signatures in the error message
            if 'CURL' in str(e) or 'HTTP' in str(e) or 'Timeout' in str(e):
                print(f"   - Network error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise # Re-raise if it's not a recognized network error
    raise RasterioIOError(f"Failed to open {href} after {max_retries} attempts.")

def carregar_banda_sentinel2_bdc(item, banda, gleba):
    """
    Load Sentinel-2 band from BDC STAC item clipped to gleba polygon.

    Parameters:
    - item: pystac Item
    - banda: str, band name like 'B04', 'SCL', etc.
    - gleba: GeoDataFrame or GeoSeries with polygon geometry (in same CRS as raster)

    Returns:
    - numpy array with clipped band data
    """
    # Get asset href for the band
    asset = item.assets.get(banda)
    if asset is None:
        raise ValueError(f"Band {banda} not found in item assets.")

    href = asset.href

    with retry_rasterio_open(href) as src:
        # Reproject gleba geometry to raster CRS if needed
        if gleba.crs != src.crs:
            gleba_proj = gleba.to_crs(src.crs)
        else:
            gleba_proj = gleba

        geoms = [feature["geometry"] for feature in gleba_proj.__geo_interface__["features"]] if hasattr(gleba_proj, "__geo_interface__") else [gleba_proj.geometry]

        # Mask raster with polygon
        out_image, out_transform = mask(src, geoms, crop=True)
        # out_image shape: (bands, height, width), Sentinel-2 bands are single band, so take first band
        band_array = out_image[0].astype(np.float32)

    return band_array

def carregar_banda_sentinel2_bdc_scl(item, banda, gleba, banda_scl_zoom):
    """
    Load Sentinel-2 band clipped to gleba and mask pixels based on SCL mask.

    Parameters:
    - item: pystac Item
    - banda: str, band name like 'B03', 'B04', etc.
    - gleba: GeoDataFrame or GeoSeries with polygon geometry
    - banda_scl_zoom: numpy array of SCL band already resized to target shape

    Returns:
    - numpy array with masked band data
    """
    banda_array = carregar_banda_sentinel2_bdc(item, banda, gleba)

    # Resize banda_array to shape of banda_scl_zoom if needed
    from skimage.transform import resize
    if banda_array.shape != banda_scl_zoom.shape:
        banda_array = resize(banda_array, banda_scl_zoom.shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    # Mask pixels where SCL indicates clouds or shadows
    # According to Sentinel-2 SCL classes:
    # 3 = Cloud Shadows, 7 = Unclassified, 8 = Cloud Medium Probability,
    # 9 = Cloud High Probability, 10 = Thin Cirrus, 11 = Snow
    cloud_classes = [3, 7, 8, 9, 10, 11]

    mask_clouds = np.isin(banda_scl_zoom, cloud_classes)
    banda_array_masked = np.where(mask_clouds, np.nan, banda_array)

    return banda_array_masked

def normalizar(array, L):
    """
    Normalize the input array to the range [0, L].

    Parameters:
    - array: numpy array of values
    - L: maximum value of the normalized scale (e.g., 4096)

    Returns:
    - normalized numpy array with values scaled to [0, L]
    """
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)
    if arr_max - arr_min == 0:
        return np.zeros_like(array)
    normalized = (array - arr_min) / (arr_max - arr_min) * L
    return normalized

def aplicar_contraste_raiz(array, L, N):
    """
    Apply root contrast enhancement to the input array.

    Parameters:
    - array: numpy array with values normalized to [0, L]
    - L: maximum value of the scale (e.g., 4096)
    - N: root degree (e.g., 2 for square root)

    Returns:
    - numpy array with contrast enhanced by applying Nth root
    """
    # Normalize array to [0,1]
    normalized = array / L
    # Apply root contrast
    contrasted = np.power(normalized, 1/N)
    # Scale back to [0, L]
    contrasted_scaled = contrasted * L
    return contrasted_scaled


# %%

#%%time
print("9. Searching for available Sentinel-2 images...")
# realiza uma busca de imagens Sentinel-2 disponíveis no
# Brazil Data Cube no intervalo temporal estabelecido
# e na região delimitada pela área de interesse (gleba)
inpe_catalog = pystac_client.Client.open(bdc_stac_link)
item_search = inpe_catalog.search(collections=[colecao_s2],
                                  datetime=f'{data_inicial}/{data_final}',
                                  bbox=gleba_4326.bounds)
time_ordered_items = sorted(list(item_search.items()),
                            key=lambda a: a.datetime,
                            reverse=False)
total_rasters = item_search.matched()

# percorrer todos os itens resultantes, obtendo sem duplicação
# as bandas 4 e 8, red e nir, para o cálculo do NDVI

delta = 1e-10  # small number to avoid division by zero

t = 0
set_of_items = set()

# calcular total de linhas de exibição, considerando um número de colunas
colunas = 4
linhas_plot = math.ceil(total_rasters / colunas) if total_rasters > 0 else 1
fig_clips_estatisticas = plt.figure(figsize=(largura_figuras, 2 * linhas_plot))
print(f"10. Processing {total_rasters} images for statistics and RGB composites...")
for item in time_ordered_items: #.items():
    tile = item.properties['bdc:tiles'][0]
    title = item.properties['datetime'][:10]
    if item.properties['datetime'] not in set_of_items:
        set_of_items.add(item.properties['datetime'])
    else:
        continue
    t = t + 1
    print(f" - Processing image {t}/{total_rasters} from {title}...")

    banda_scl = carregar_banda_sentinel2_bdc(item, 'SCL', gleba)
    banda_04 = carregar_banda_sentinel2_bdc(item, 'B04', gleba)
    banda_scl_zoom = resize(banda_scl, (banda_04.shape[0], banda_04.shape[1]), order=0)
    banda_03 = carregar_banda_sentinel2_bdc_scl(item, 'B03', gleba, banda_scl_zoom)
    banda_04 = carregar_banda_sentinel2_bdc_scl(item, 'B04', gleba, banda_scl_zoom)
    banda_06 = carregar_banda_sentinel2_bdc_scl(item, 'B06', gleba, banda_scl_zoom)
    banda_07 = carregar_banda_sentinel2_bdc_scl(item, 'B07', gleba, banda_scl_zoom)
    banda_08 = carregar_banda_sentinel2_bdc_scl(item, 'B08', gleba, banda_scl_zoom)
    banda_11 = carregar_banda_sentinel2_bdc(item, 'B11', gleba)
    banda_11_zoom = resize(banda_11, (banda_04.shape[0], banda_04.shape[1]), order=0)
    banda_12 = carregar_banda_sentinel2_bdc(item, 'B12', gleba)
    banda_12_zoom = resize(banda_12, (banda_04.shape[0], banda_04.shape[1]), order=0)

    # utilizar metadados SCL (Scene Classification Map) para auxiliar a visualização
    total_pixels = len(banda_scl_zoom[banda_scl_zoom > 0])
    total_nuvens = len(banda_scl_zoom[banda_scl_zoom > 7])
    porcentagem_nuvens = 100 * total_nuvens / total_pixels

    # vegetação baseado no NDVI
    limiar_vegetacao = 0.7
    banda_ndvi = (banda_08 - banda_04) / (banda_08 + banda_04 + delta)
    ndvi = banda_ndvi[np.isfinite(banda_ndvi)]
    total_vegetacao = len(ndvi[ndvi > limiar_vegetacao])
    porcentagem_vegetacao = 100 * total_vegetacao / total_pixels
    
    # áreas queimadas baseado no NBR
    limiar_queimados = -0.45
    banda_nbr = (banda_08 - banda_11_zoom) / (banda_08 + banda_11_zoom + delta)
    nbr = banda_nbr[np.isfinite(banda_nbr)]
    total_queimados = len(nbr[nbr < limiar_queimados])
    porcentagem_queimados = 100 * total_queimados / total_pixels

    # a textura está sendo calculada como o desvio padrão dos valores
    # dos pixels da banda NDVI, em valor absoluto
    textura = np.absolute(np.nanstd(ndvi) * 100)

    # normalized difference water index
    limiar_umidade = -0.1
    banda_ndwi = (banda_03 - banda_08) / (banda_03 + banda_08)
    ndwi = banda_ndwi[np.isfinite(banda_ndwi)]
    total_umidade = len(ndwi[ndwi > limiar_umidade])
    porcentagem_umidade = 100 * total_umidade / total_pixels

    # Burned Area Index for Sentinel-2
    # esse índice poderá ser utilizado de forma complementar ao NBR
    banda_bais = (1 - np.sqrt(banda_06 + banda_07 + banda_08)/banda_04) * \
                 ((banda_12_zoom - banda_08)/np.sqrt(banda_12_zoom + banda_08) + 1)

    # Soil Composition Index
    limiar_solo = 0.0
    banda_sci = (banda_11_zoom - banda_08) / (banda_11_zoom + banda_08)
    sci = banda_sci[np.isfinite(banda_sci)]
    total_solo = len(sci[sci > limiar_solo])
    porcentagem_solo = 100 * total_solo / total_pixels
    
    # (3 x linhas x colunas)
    matriz_rgb = np.zeros((banda_04.shape[0], banda_04.shape[1], 3))
    
    # a quantização do Sentinel-2 é de 12 bits, ou seja,
    # os valores dos números digitais variam entre 0 e 2^12 = 4096
    L = 2**12 
    N = 2
    matriz_normalizada_vermelho = normalizar(banda_08, L)
    matriz_normalizada_verde = normalizar(banda_11_zoom, L)
    matriz_normalizada_azul = normalizar(banda_04, L)
    
    matriz_raiz_vermelho = aplicar_contraste_raiz(matriz_normalizada_vermelho, L, N)
    matriz_raiz_verde = aplicar_contraste_raiz(matriz_normalizada_verde, L, N)
    matriz_raiz_azul = aplicar_contraste_raiz(matriz_normalizada_azul, L, N)
    
    matriz_rgb[:, :, 0] = matriz_raiz_vermelho / L
    matriz_rgb[:, :, 1] = matriz_raiz_verde / L
    matriz_rgb[:, :, 2] = matriz_raiz_azul / L

    # versão com barras de nuvens/vegetação
    plt.subplot(linhas_plot, 2 * colunas, 2 * t - 1)
    plt.imshow(matriz_rgb)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    plt.subplot(linhas_plot, 2 * colunas, 2 * t)
    plt.barh([1, 2, 3, 4, 5, 6], 
             [porcentagem_nuvens, porcentagem_vegetacao, porcentagem_queimados, textura, porcentagem_umidade, porcentagem_solo], 
             # tick_label=['nuvens', 'vegetacao', 'queimados', 'textura', 'agua', 'solo'],
             tick_label=['nv', 'vg', 'qm', 'tx', 'um', 'so'],
             color=['tab:cyan', 'tab:green', 'tab:orange', 'tab:red', 'tab:blue', 'tab:brown']) 
    plt.xlim([0, 100])
    plt.box(False)
    plt.tight_layout()
    plt.title(f'{title}');

# %%
#%%time
# This cell calculates and plots the Enhanced Vegetation Index (EVI) for each available image.

print("11. Processing images for EVI calculation...")
# Define EVI constants for Sentinel-2
G = 2.5
C1 = 6.0
C2 = 7.5
L_evi = 1.0
SCALE_FACTOR = 10000.0 # To convert DN to reflectance

# Setup for plotting
colunas_evi = 4
linhas_plot_evi = math.ceil(total_rasters / colunas_evi) if total_rasters > 0 else 1
fig_evi_clips = plt.figure(figsize=(largura_figuras, 3 * linhas_plot_evi))
plt.suptitle('EVI (Enhanced Vegetation Index) for each date', fontsize=16)

t_evi = 0
set_of_items_evi = set()

for item in time_ordered_items:
    if item.properties['datetime'] in set_of_items_evi:
        continue
    set_of_items_evi.add(item.properties['datetime'])
    t_evi += 1
    print(f" - Calculating EVI for image {t_evi}/{total_rasters} from {item.properties['datetime'][:10]}...")

    try:
        # Load required bands for EVI (B02, B04, B08) and SCL for masking
        banda_scl = carregar_banda_sentinel2_bdc(item, 'SCL', gleba)
        
        # Use B04 shape as reference
        banda_04_ref = carregar_banda_sentinel2_bdc(item, 'B04', gleba)
        if banda_04_ref.size == 0:
            print(f"Skipping {item.id} due to empty B04 band.")
            continue
            
        banda_scl_zoom = resize(banda_scl, banda_04_ref.shape, order=0, preserve_range=True, anti_aliasing=False)

        # Load bands with cloud masking
        banda_02 = carregar_banda_sentinel2_bdc_scl(item, 'B02', gleba, banda_scl_zoom)
        banda_04 = carregar_banda_sentinel2_bdc_scl(item, 'B04', gleba, banda_scl_zoom)
        banda_08 = carregar_banda_sentinel2_bdc_scl(item, 'B08', gleba, banda_scl_zoom)

        # Convert to float and scale to reflectance
        refl_02 = banda_02.astype(np.float32) / SCALE_FACTOR
        refl_04 = banda_04.astype(np.float32) / SCALE_FACTOR
        refl_08 = banda_08.astype(np.float32) / SCALE_FACTOR

        # Calculate EVI
        numerator = G * (refl_08 - refl_04)
        denominator = (refl_08 + C1 * refl_04 - C2 * refl_02 + L_evi)
        banda_evi = np.divide(numerator, denominator, where=denominator!=0)
        
        # Plotting
        ax = plt.subplot(linhas_plot_evi, colunas_evi, t_evi)
        im = ax.imshow(banda_evi, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(item.properties['datetime'][:10])
        ax.set_xticks([])
        ax.set_yticks([])

    except Exception as e:
        print(f"Could not process item {item.id}: {e}")
        # Plot an empty subplot to keep alignment
        ax = plt.subplot(linhas_plot_evi, colunas_evi, t_evi)
        ax.set_title(f"{item.properties['datetime'][:10]}\n(Error)")
        ax.set_xticks([])
        ax.set_yticks([])

# Add a colorbar for the EVI plot
fig_evi_clips.subplots_adjust(right=0.9)
cbar_ax = fig_evi_clips.add_axes([0.92, 0.15, 0.02, 0.7])
fig_evi_clips.colorbar(im, cax=cbar_ax, label='EVI')

plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for colorbar
plt.show()

# %% [markdown]
# ## Geração de agrupamentos das séries dentro da gleba por meio do SOM (*Self Organizing Maps*)
# ---

# %%
import pystac_client
import rioxarray
import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import mapping
import dask.diagnostics
from odc.stac import stac_load

def process_cube(shapefile, query_bands, start_date, end_date, collections, stac_url, resolution=10):
    """
    Load and preprocess a data cube clipped to the shapefile polygon.
    This version is optimized for memory usage using odc-stac.
    It creates a gap-filled monthly maximum value composite.

    Parameters:
    - shapefile: GeoDataFrame or GeoSeries with polygon geometry (in EPSG:4326)
    - query_bands: list of band names to load (e.g., ['NDVI', 'SCL'])
    - start_date, end_date: strings 'YYYY-MM-DD'
    - collections: list of STAC collection names
    - stac_url: STAC API URL string
    - resolution: int, the target resolution for the output cube.

    Returns:
    - cube: xarray.Dataset with dimensions (time, y, x) containing the processed NDVI band.
            The cube is gap-filled and ready for analysis.
    - spatial_mask: xarray.DataArray boolean mask of valid pixels within the polygon.
    """

    print("   - Opening STAC catalog and searching for items...")
    catalog = pystac_client.Client.open(stac_url)
    
    # Use the bounding box from the shapefile for the STAC search
    search_bbox = shapefile.total_bounds.tolist()
    
    items = catalog.search(
        collections=collections,
        bbox=search_bbox,
        datetime=f"{start_date}/{end_date}"
    ).get_all_items()

    filtered_items = [item for item in items if all(band in item.assets for band in query_bands)]
    if not filtered_items:
        raise RuntimeError("No items found with all requested bands.")
    
    print(f"   - Found {len(filtered_items)} items. Loading data cube with odc-stac...")
    
    # Use odc-stac to load the data cube. It handles reprojection, alignment, and lazy loading.
    # We provide the geometry of the area of interest to clip the data.
    geom = shapefile.geometry.iloc[0]
    
    # Determine the appropriate UTM CRS for the area of interest
    utm_crs = shapefile.estimate_utm_crs()
    print(f"   - Using projected CRS: {utm_crs.to_string()}")

    # Trigger computation with a progress bar
    with dask.diagnostics.ProgressBar():
        cube = stac_load(
            filtered_items,
            bands=query_bands,
            crs=utm_crs,
            resolution=resolution,
            geopolygon=geom, # Use geopolygon for clipping
            resampling={"SCL": "nearest", "*": "bilinear"}, # Use nearest for SCL, bilinear for others
            chunks={"x": 2048, "y": 2048}, # Use dask chunks for memory efficiency
        )
    
    # --- DEBUGGING: Print the cube object structure ---
    print("   - DEBUG: Cube object structure after loading:")
    print(cube)
    # --- END DEBUGGING ---

    # Apply scale factor to NDVI band to convert from integer to float [-1, 1]
    if 'NDVI' in cube:
        print("   - Applying scale factor to NDVI band...")
        cube['NDVI'] = cube['NDVI'] / 10000.0

    # Create mask from the original SCL band BEFORE interpolation.
    # A pixel is considered valid if it's not cloud/shadow in at least one image.
    if 'SCL' in cube:
        # Valid SCL classes for vegetation analysis
        valid_scl_codes = [4, 5, 6]  # 4: Vegetation, 5: Not Vegetated, 6: Water

        # --- DEBUGGING: Print SCL band info ---
        # This part is wrapped in a rasterio.Env to ensure network resiliency for the compute() call
        with rasterio.Env(GDAL_HTTP_RETRY=5, GDAL_HTTP_RETRY_DELAY=10, GDAL_HTTP_TIMEOUT=120):
            scl_band = cube['SCL']
            unique_values = np.unique(scl_band.compute())
            print(f"   - DEBUG: Unique SCL values found in cube: {unique_values}")

        # Create a spatial mask: a pixel is valid if it ever has a valid SCL code.
        spatial_mask = cube['SCL'].isin(valid_scl_codes).any(dim='time')
        with rasterio.Env(GDAL_HTTP_RETRY=5, GDAL_HTTP_RETRY_DELAY=10, GDAL_HTTP_TIMEOUT=120):
            print(f"   - DEBUG: Total valid pixels in mask: {spatial_mask.sum().compute().item()}")
    else:
        # If no SCL band, assume all pixels are valid initially.
        spatial_mask = xr.ones_like(cube[query_bands[0]].isel(time=0), dtype=bool)

    print("   - Creating monthly maximum value composites...")
    # Before resampling, set data to NaN where SCL indicates invalid pixels
    # This ensures they are ignored by the .max() operation.
    if 'SCL' in cube:
        # All codes other than the valid ones are considered invalid for the composite
        invalid_scl_codes = [0, 1, 2, 3, 7, 8, 9, 10, 11]
        cube = cube.where(cube['SCL'].isin(valid_scl_codes))
    
    # Resample to monthly frequency, taking the maximum value in each month.
    # This creates a Maximum Value Composite (MVC).
    monthly_cube = cube.resample(time='1M').max(skipna=True)

    print("   - Imputing missing monthly values via interpolation...")
    # Interpolate linearly along time to fill gaps left after MVC.
    # Rechunk along time so interpolation can work across the entire series.
    # Then back-fill and forward-fill to handle start/end NaNs.
    monthly_cube = monthly_cube.chunk({"time": -1}).interpolate_na(dim='time', method='linear')
    monthly_cube = monthly_cube.bfill(dim='time').ffill(dim='time')

    # Drop the SCL band as it's no longer needed, and compute the final result.
    final_cube = monthly_cube.drop_vars('SCL', errors='ignore')
    
    # Compute the final cube and the mask separately within a resilient environment
    with rasterio.Env(GDAL_HTTP_RETRY=5, GDAL_HTTP_RETRY_DELAY=10, GDAL_HTTP_TIMEOUT=120):
        return final_cube.compute(), spatial_mask.compute()

# !pip install minisom
import numpy as np
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

def som_time_series_clustering(cube, mask, n=2, random_seed=123, n_parallel=0, training_steps=100):
    """
    Perform SOM clustering on time series data cube.

    Parameters:
    - cube: xarray.Dataset with dimensions (time, y, x, band)
    - mask: xarray.DataArray boolean mask of valid pixels
    - n: int, SOM grid size (n x n)
    - random_seed: int, random seed for reproducibility
    - n_parallel: int, number of parallel jobs (not used here)
    - training_steps: int, number of SOM training iterations

    Returns:
    - result: 2D numpy array (y, x) with cluster indices
    - neuron_weights: numpy array (n*n, time*bands) with SOM codebooks
    - predictions: 2D numpy array (y, x) with neuron indices
    - som: The trained MiniSom object.
    """

    # Work directly with the NDVI DataArray
    cube_da = cube['NDVI']

    # Reshape data for clustering: (pixels, time*bands)
    # The input is (time, y, x). We want (y, x, time)
    data = cube_da.transpose('y', 'x', 'time').values
    pixel_count = data.shape[0] * data.shape[1]
    feature_count = data.shape[2]
    data = data.reshape(pixel_count, feature_count)

    # Apply mask to select valid pixels
    if mask is not None:
        # The mask is now a 2D spatial mask
        mask_flat = mask.values.flatten()
        data_valid = data[mask_flat]
    else:
        mask_flat = np.ones(pixel_count, dtype=bool)
        data_valid = data


    # Check if there is any data to process
    if data_valid.shape[0] == 0:
        raise ValueError("No valid data points to cluster after masking.")

    # Normalize data
    print("   - Normalizing time series data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_valid)

    # Initialize and train SOM
    print("   - Training SOM...")
    som = MiniSom(n, n, data_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=random_seed)
    som.random_weights_init(data_scaled)
    som.train_random(data_scaled, training_steps)

    # Get neuron weights (codebooks) and inverse transform them
    neuron_weights_scaled = som.get_weights().reshape(n*n, -1)
    neuron_weights = scaler.inverse_transform(neuron_weights_scaled)

    # Predict cluster for each valid pixel
    # The SOM was trained on scaled data, so prediction must also use scaled data.
    winners = np.array([som.winner(x) for x in data_scaled])
    predictions_valid = winners[:, 0] * n + winners[:, 1]

    # Create full prediction array with -1 for masked pixels
    predictions = -1 * np.ones(mask_flat.shape, dtype=int)
    predictions[mask_flat] = predictions_valid
    predictions = predictions.reshape(mask.shape) # Reshape to (y, x)

    # Result can be the same as predictions or aggregated
    result = predictions

    return result, neuron_weights, predictions, som

import matplotlib.pyplot as plt

def plot_som_u_matrix(som, cmap='viridis_r'):
    """
    Plots the U-Matrix of a trained SOM with cluster numbers overlaid.

    The U-Matrix visualizes the distances between neighboring neurons,
    helping to identify cluster boundaries. Cool colors (valleys) represent
    well-defined clusters, while hot colors (ridges) represent boundaries.

    Parameters:
    - som: A trained MiniSom object.
    - cmap: The colormap to use for the distance visualization. A reversed
            map like 'viridis_r' is often intuitive (cool=low distance).

    Returns:
    - fig, ax: matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    u_matrix = som.distance_map()
    som_x, som_y = som.get_weights().shape[:2]

    # Use a path effect to add a black stroke to the text for readability
    path_effects = [withStroke(linewidth=3, foreground='black')]

    # Determine a good text color based on the U-Matrix values
    # This is a simple heuristic: use white for dark cells, black for light cells
    text_colors = np.empty(u_matrix.shape, dtype=object)
    mid_point = (np.nanmax(u_matrix) + np.nanmin(u_matrix)) / 2
    text_colors[u_matrix < mid_point] = 'white'
    text_colors[u_matrix >= mid_point] = 'black'
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(u_matrix, cmap=cmap)
    ax.set_title('SOM U-Matrix (Cluster Boundaries)')
    
    # Add a colorbar to show distance scale
    plt.colorbar(im, ax=ax, label='Distance between neurons')
    plt.tight_layout()

    # Add cluster numbers to each cell
    for i in range(som_x):
        for j in range(som_y):
            cluster_index = i * som_y + j
            ax.text(j, i, str(cluster_index),
                    ha='center', va='center',
                    color=text_colors[i, j],
                    fontweight='bold',
                    path_effects=path_effects)
    return fig, ax

def plot_cluster_profiles(cube, predictions, band_name, cmap):
    """
    Calculates and plots the mean time series for each cluster.

    Parameters:
    - cube: xarray.Dataset with the original time series data (e.g., monthly NDVI).
    - predictions: 2D numpy array (y, x) with the final cluster indices for each pixel.
    - band_name: str, band to plot (e.g., 'NDVI')
    - cmap: The colormap used for the clusters.

    Returns:
    - fig, ax matplotlib figure and axes
    """
    
    # Ensure we are working with the DataArray
    da = cube[band_name]
    time_coords = da['time'].values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find the unique cluster IDs, ignoring the -1 mask value
    cluster_ids = np.unique(predictions[predictions >= 0])

    for cluster_id in cluster_ids:
        # Create a boolean mask for the current cluster
        cluster_mask = (predictions == cluster_id)
        
        # Use the mask to select the relevant time series from the DataArray and calculate the mean
        mean_series = da.where(cluster_mask).mean(dim=['x', 'y'])
        
        ax.plot(time_coords, mean_series, label=f'Cluster {cluster_id}', color=cmap(cluster_id))

    ax.set_title(f'Mean NDVI Time Series Profiles for each SOM Cluster')
    ax.set_xlabel('Time')
    ax.set_ylabel(band_name)
    ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_cluster_image(predictions, cmap):
    """
    Plot spatial cluster assignments.

    Parameters:
    - predictions: 2D array with cluster indices
    - cmap: The colormap to use for clusters.

    Returns:
    - fig, ax matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(largura_figuras, largura_figuras * 0.8))
    # Handle case where no clusters were assigned (-1)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)

    # Use BoundaryNorm for a discrete colormap that maps integers to colors correctly.
    # The boundaries are set at the midpoints between the integers.
    bounds = np.arange(min_pred, max_pred + 2) - 0.5
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Create a masked array to show -1 as a specific color (e.g., white or gray)
    masked_preds = np.ma.masked_equal(predictions, -1)
    # Use a copy of the colormap to avoid modifying the original
    cmap_copy = cmap
    cmap_copy.set_bad('white', 1.)

    im = ax.imshow(masked_preds, cmap=cmap_copy, norm=norm, interpolation='none')
    ax.set_title('SOM Clustering Result')

    # Adjust colorbar to show correct ticks
    ticks = np.arange(min_pred + 1, max_pred + 1)
    plt.colorbar(im, ax=ax, ticks=ticks)
    plt.tight_layout()
    return fig, ax

def plot_som_neuron_map(som, cmap):
    """
    Plots the SOM neurons as a hexagonal grid, colored by cluster index.

    Parameters:
    - som: A trained MiniSom object.
    - cmap: The colormap to use for coloring the neurons.

    Returns:
    - fig, ax: matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('SOM Neuron Map')

    som_x, som_y = som.get_weights().shape[:2]

    # Create hexagonal grid positions
    for i in range(som_x):
        for j in range(som_y):
            # Calculate center of hexagon
            center_x = i + 0.5 * (j % 2)
            center_y = j * np.sqrt(3) / 2
            
            # Get cluster index and color
            cluster_index = i * som_y + j
            color = cmap(cluster_index)
            
            # Create hexagon
            hexagon = mpatches.RegularPolygon((center_x, center_y), numVertices=6, radius=0.5,
                                              facecolor=color, edgecolor='k')
            ax.add_patch(hexagon)
            
            # Add text label for the cluster
            ax.text(center_x, center_y, str(cluster_index), ha='center', va='center', color='white',
                    fontweight='bold')

    ax.autoscale_view()
    plt.tight_layout()
    return fig, ax


# from geo_credito_rural_utils_v2 import *

print("12. Creating data cube for SOM clustering...")
cubo, mask = process_cube(shapefile=gleba,
                        query_bands = ['NDVI', 'SCL'],
                        start_date = data_inicial,
                        end_date = data_final,
                        collections = [colecao_s2],
                        stac_url = bdc_stac_link,
                        resolution=10)

# Crop the cube to the exact geometry using salem and fill NaNs
print("   - Cropping data cube to exact geometry with rioxarray...")
# The 'gleba' GeoDataFrame needs to be in the same CRS as the cube.
utm_crs = gleba.estimate_utm_crs()
gleba_proj = gleba.to_crs(utm_crs)

# Explicitly set the CRS and clip using rioxarray
# cubo.rio.write_crs(utm_crs, inplace=True)
# cubo = cubo.rio.clip(geometries=gleba_proj.geometry, all_touched=True, from_disk=True)

# --- DEBUGGING: Plot average NDVI time series ---
print("   - Plotting average NDVI time series for debugging...")
fig_avg_ndvi, ax_avg_ndvi = plt.subplots(figsize=(12, 5))
average_ndvi = cubo['NDVI'].mean(dim=['x', 'y'])
average_ndvi.plot(ax=ax_avg_ndvi, marker='o')
ax_avg_ndvi.set_title('Average Monthly NDVI for the Entire Property')
ax_avg_ndvi.set_xlabel('Time')
ax_avg_ndvi.set_ylabel('Average NDVI')
ax_avg_ndvi.grid(True)
plt.tight_layout()

# Instead of plt.show()
fig_avg_ndvi.savefig("output/debug_avg_ndvi.png")
print("   - Average NDVI plot saved to output/debug_avg_ndvi.png")

# cubo = cubo.fillna(-1)

# --- END DEBUGGING ---

# %%

# o parâmetro 'n' indica que o método vai criar uma rede
# de n x n neurônios e tentar agrupar a série em n x n clusters
sqrt_n = 3

print("13. Performing SOM clustering...")
result, neuron_weights, predictions, som = som_time_series_clustering(cubo, mask,
                                                                      n = sqrt_n, random_seed = 123,
                                                                      n_parallel = 0, training_steps = 100)

# Create a discrete colormap to ensure colors match between plots.
num_clusters = sqrt_n * sqrt_n
base_cmap = plt.get_cmap('tab20', num_clusters)
color_list = [base_cmap(i) for i in range(num_clusters)]
cmap = matplotlib.colors.ListedColormap(color_list, name='som_clusters')

fig_som_profiles, _ = plot_cluster_profiles(cubo, predictions,
                                            band_name='NDVI',
                                            cmap=cmap)

# %%
fig_som_neuron_map, _ = plot_som_neuron_map(som, cmap)

# %%
fig_som_u_matrix, _ = plot_som_u_matrix(som)

# %%
fig_som_clustering, _ = plot_cluster_image(predictions, cmap)

# %% [markdown]
# ## Finalização de relatório com as saídas obtidas
# ---

# %%
def create_title_page(title, pdf_pages):
    """Creates a title page figure and saves it to the PDF."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.5, 0.5, title, ha='center', va='center', fontsize=24, wrap=True)
    pdf_pages.savefig(fig)
    plt.close(fig)

print("14. Generating PDF report...")
# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
arquivo_relatorio = os.path.join(output_dir, f'gleba-refbacen-{ref_bacen}_{data_inicial}-a-{data_final}.pdf')
relatorio = PdfPages(arquivo_relatorio)

try:
    # Add the new localization map at the beginning
    create_title_page("Mapa de localização da Gleba", relatorio)
    relatorio.savefig(fig_localization, bbox_inches='tight')

    create_title_page("Visualização de Pontos Aleatórios na Gleba", relatorio)
    relatorio.savefig(fig_pontos_aleatorios, bbox_inches='tight')

    create_title_page("Cruzamento com MapBiomas", relatorio)
    relatorio.savefig(fig_mapbiomas, bbox_inches='tight')

    create_title_page("Séries Temporais (NDVI)", relatorio)
    relatorio.savefig(fig_time_series, bbox_inches='tight')

    create_title_page("Métricas Fenológicas para Pontos Aleatórios", relatorio)
    for fig_phenometrics in vetor_figs_phenometrics:
        relatorio.savefig(fig_phenometrics, bbox_inches='tight')

    create_title_page("Recortes de Imagens e Estatísticas Temporais", relatorio)
    relatorio.savefig(fig_clips_estatisticas, bbox_inches='tight')

    create_title_page("Índice de Vegetação Realçado (EVI)", relatorio)
    if 'fig_evi_clips' in locals():
        relatorio.savefig(fig_evi_clips, bbox_inches='tight')

    create_title_page("Agrupamento por Mapas Auto-Organizáveis (SOM)", relatorio)
    relatorio.savefig(fig_som_profiles, bbox_inches='tight')
    relatorio.savefig(fig_som_neuron_map, bbox_inches='tight')
    relatorio.savefig(fig_som_u_matrix, bbox_inches='tight')
    relatorio.savefig(fig_som_clustering, bbox_inches='tight')
finally:
    plt.close('all') # Close all figures to free up memory

metadados = relatorio.infodict()
metadados['Title'] = f'Relatório sobre gleba {ref_bacen}, entre {data_inicial} e {data_final}'
metadados['Author'] = 'Geo Crédito Rural'
metadados['Subject'] = 'Cruzamento com séries temporais e estatísticas'
metadados['Keywords'] = 'gleba, crédito rural, proagro, sicor, sentinel, cbers'
metadados['CreationDate'] = datetime.today()

relatorio.close()

print(f'Arquivo {arquivo_relatorio} salvo.')

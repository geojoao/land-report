"""
Enhanced script to test and analyze phenophase detection on test geometries.
Loads geometries from test_geometries.csv and visualizes NDVI time series
with detailed phenophase stage detection analysis.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from shapely import wkt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Add current directory to path to import test_wtss functions
sys.path.insert(0, '/workspaces/land-report')

from test_wtss import get_and_plot_areal_ts_wtss
from phenophase import extract_phenometrics

def analyze_and_report(geometry_wkt, geom_id):
    """Analyze phenophase detection and provide detailed report."""
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS FOR GEOMETRY {geom_id}")
    print(f"{'='*80}\n")
    
    try:
        # Parse geometry
        gleba_4326 = wkt.loads(geometry_wkt)
        
        # Get time series
        import wtss
        from scipy.signal import savgol_filter
        
        bdc_wtss_link = 'https://data.inpe.br/bdc/wtss/v4/'
        colecao_modis = 'mod13q1-6.1'
        servico_wtss = wtss.WTSS(bdc_wtss_link)
        cubo = servico_wtss[colecao_modis]
        
        end_date_dt = datetime.now()
        start_date_dt = datetime(2020, 1, 1)
        start_date_str = start_date_dt.strftime('%Y-%m-%d')
        end_date_str = end_date_dt.strftime('%Y-%m-%d')
        
        gleba_geojson = gleba_4326.__geo_interface__
        timeseries = cubo.ts(
            attributes=('NDVI',),
            geom=gleba_geojson,
            start_date=start_date_str,
            end_date=end_date_str,
        )
        
        summarize = timeseries.summarize()
        df = summarize.df()
        
        # Convert to wide format
        df['band_agg'] = df['attribute'] + '_' + df['aggregation']
        wide_df = df.pivot(index='datetime', columns='band_agg', values='value')
        wide_df = wide_df.reset_index()
        wide_df.columns.name = None
        df_ts = wide_df
        
        # Apply smoothing
        for col in df_ts.columns:
            if col != 'datetime' and pd.api.types.is_numeric_dtype(df_ts[col]):
                if len(df_ts[col]) > 5:
                    df_ts[col] = savgol_filter(df_ts[col], window_length=5, polyorder=2)
        
        print(f"Time series data:")
        print(f"  - Date range: {df_ts['datetime'].min()} to {df_ts['datetime'].max()}")
        print(f"  - Number of observations: {len(df_ts)}")
        print(f"  - NDVI mean stats: min={df_ts['NDVI_mean'].min():.3f}, max={df_ts['NDVI_mean'].max():.3f}, mean={df_ts['NDVI_mean'].mean():.3f}")
        print(f"  - NDVI std: {df_ts['NDVI_mean'].std():.3f}")
        
        # Test phenophase detection with different parameters
        print(f"\n{'─'*80}")
        print("Testing phenophase detection with different configurations:")
        print(f"{'─'*80}\n")
        
        configs = [
            {
                'name': 'Current (conservative)',
                'min_cycle_length_days': 45,
                'quality_threshold': 0.55,
                'smoothing_method': 'both',
                'quantile_trough': 25
            },
            {
                'name': 'More sensitive',
                'min_cycle_length_days': 35,
                'quality_threshold': 0.50,
                'smoothing_method': 'both',
                'quantile_trough': 30
            },
            {
                'name': 'Very sensitive (for short cycles)',
                'min_cycle_length_days': 30,
                'quality_threshold': 0.45,
                'smoothing_method': 'both',
                'quantile_trough': 35
            },
        ]
        
        results = {}
        for config in configs:
            print(f"Config: {config['name']}")
            phenometrics = extract_phenometrics(
                df_ts, 
                ndvi_column='NDVI_mean',
                min_cycle_length_days=config['min_cycle_length_days'],
                quality_threshold=config['quality_threshold'],
                smoothing_method=config['smoothing_method'],
                quantile_trough=config['quantile_trough']
            )
            
            if phenometrics.get('success', False):
                cycles = phenometrics['cycles']
                successful = [c for c in cycles if c.get('fit_success', False)]
                print(f"  ✓ Total cycles detected: {phenometrics['num_cycles_detected']}")
                print(f"  ✓ Successful fits: {phenometrics['num_successful_fits']}")
                print(f"  ✓ Mean R²: {phenometrics['mean_r_squared']:.4f}")
                
                # Analyze cycles
                if successful:
                    print(f"  ✓ Cycle details:")
                    for i, cycle in enumerate(successful[:5]):  # Show first 5
                        sos = cycle['phenophase_dates']['sos']
                        pos = cycle['phenophase_dates']['pos']
                        eos = cycle['phenophase_dates']['eos']
                        cycle_length = (eos - sos).days
                        print(f"    - Cycle {i+1}: {sos.strftime('%Y-%m-%d')} to {eos.strftime('%Y-%m-%d')} ({cycle_length} days)")
                        
                        # Check for double cropping pattern
                        if i > 0:
                            prev_eos = successful[i-1]['phenophase_dates']['eos']
                            gap = (sos - prev_eos).days
                            if gap < 60:
                                print(f"      ⚠️  Gap from previous cycle: {gap} days (potential double cropping)")
            else:
                print(f"  ✗ No cycles detected")
            
            print()
            results[config['name']] = phenometrics
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Load geometries and test phenophase detection on each."""
    
    print("="*80)
    print("PHENOPHASE DETECTION ANALYSIS FOR TEST GEOMETRIES")
    print("="*80)
    
    # Load geometries from CSV
    print("\nLoading geometries from test_geometries.csv...")
    geometries_df = pd.read_csv('/workspaces/land-report/test_geometries.csv', index_col=0)
    print(f"Found {len(geometries_df)} geometries to test.\n")
    
    # Create output directory for images
    output_dir = '/workspaces/land-report/phenophase_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each geometry
    all_results = {}
    for geom_id, row in geometries_df.iterrows():
        geometry_wkt = row['geometry']
        geom_id_str = f'Geometry_{geom_id}'
        
        # Perform detailed analysis
        results = analyze_and_report(geometry_wkt, geom_id)
        all_results[geom_id_str] = results
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

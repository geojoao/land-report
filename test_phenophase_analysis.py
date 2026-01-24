"""
Script to test and analyze phenophase detection on test geometries.
Loads geometries from test_geometries.csv and visualizes NDVI time series
with phenophase stage detection.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from shapely import wkt
from matplotlib.backends.backend_pdf import PdfPages

# Add current directory to path to import test_wtss functions
sys.path.insert(0, '/workspaces/land-report')

from test_wtss import get_and_plot_areal_ts_wtss

def main():
    """Load geometries and test phenophase detection on each."""
    
    # Load geometries from CSV
    print("Loading geometries from test_geometries.csv...")
    geometries_df = pd.read_csv('/workspaces/land-report/test_geometries.csv', index_col=0)
    print(f"Found {len(geometries_df)} geometries to test.\n")
    
    # Create PDF to save all plots
    pdf_path = '/workspaces/land-report/phenophase_analysis.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Iterate through each geometry
        for geom_id, row in geometries_df.iterrows():
            try:
                print(f"\n{'='*70}")
                print(f"Processing Geometry {geom_id}")
                print(f"{'='*70}")
                
                # Parse WKT geometry
                geometry_wkt = row['geometry']
                gleba_4326 = wkt.loads(geometry_wkt)
                
                # Print geometry info
                bounds = gleba_4326.bounds
                print(f"Geometry bounds: {bounds}")
                print(f"Geometry type: {gleba_4326.geom_type}")
                
                # Call the function to get and plot the time series
                gleba_id = f'Geometry_{geom_id}'
                
                # No specific planting/harvesting periods - let the function detect them
                fig = get_and_plot_areal_ts_wtss(
                    gleba_4326, 
                    gleba_id, 
                    planting_period=None, 
                    harvesting_period=None
                )
                
                # Save figure to PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                print(f"✓ Successfully processed Geometry {geom_id}")
                
            except Exception as e:
                print(f"✗ Error processing Geometry {geom_id}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! PDF saved to: {pdf_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 1.4: Identify the Functional Enhancer

This script analyzes Enformer prediction results to identify candidate enhancers
affected by the target SNP.

Usage:
    python identify_functional_enhancer.py --config ../config/config.ini --target-snp rs1000000
"""

import argparse
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import h5py
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancerIdentifier:
    def __init__(self, config_file, target_snp=None):
        """Initialize enhancer identifier with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.disease_name = self.config['disease']['name']
        self.output_dir = Path(self.config['data']['output_dir'])
        
        # Analysis parameters
        self.effect_threshold = float(self.config['analysis']['effect_size_threshold'])
        
        self.target_snp = target_snp
        
    def load_enformer_results(self):
        """Load Enformer prediction results from Phase 1.3."""
        if self.target_snp is None:
            # Find the most recent Enformer results
            h5_files = list(self.output_dir.glob("*_enformer_predictions.h5"))
            if not h5_files:
                raise FileNotFoundError("No Enformer prediction files found")
            h5_file = max(h5_files, key=lambda x: x.stat().st_mtime)
            self.target_snp = h5_file.stem.replace('_enformer_predictions', '')
        else:
            h5_file = self.output_dir / f"{self.target_snp}_enformer_predictions.h5"
            
        if not h5_file.exists():
            raise FileNotFoundError(f"Enformer predictions not found: {h5_file}")
            
        logger.info(f"Loading Enformer results from: {h5_file}")
        
        results = {}
        with h5py.File(h5_file, 'r') as f:
            results['reference_predictions'] = f['reference_predictions'][:]
            results['variant_predictions'] = f['variant_predictions'][:]
            
            # Load track-specific results
            for track_name in f.keys():
                if track_name not in ['reference_predictions', 'variant_predictions']:
                    track_data = {}
                    for key in f[track_name].keys():
                        track_data[key] = f[track_name][key][:]
                    results[track_name] = track_data
                    
        # Load summary metadata
        summary_file = self.output_dir / f"{self.target_snp}_enformer_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                results['metadata'] = json.load(f)
                
        return results
        
    def calculate_genomic_coordinates(self, bin_indices, sequence_metadata):
        """Calculate genomic coordinates for Enformer output bins."""
        # Enformer outputs 896 bins, each representing ~128bp resolution
        bin_size = 128  # bp per bin
        center_bin = 896 // 2  # Middle bin
        
        # SNP position
        snp_pos = sequence_metadata['position']
        
        # Calculate start position of each bin
        bin_positions = []
        for bin_idx in bin_indices:
            # Distance from center bin
            distance_from_center = (bin_idx - center_bin) * bin_size
            bin_start = snp_pos + distance_from_center
            bin_end = bin_start + bin_size - 1
            bin_positions.append({
                'bin_index': bin_idx,
                'chr': sequence_metadata['chromosome'],
                'start': bin_start,
                'end': bin_end,
                'center': bin_start + bin_size // 2
            })
            
        return bin_positions
        
    def identify_significant_regions(self, track_data, track_name):
        """Identify genomically significant regions for a specific track."""
        log2_fc = track_data['log2_fold_change']
        
        # Calculate statistics across all bins
        abs_log2_fc = np.abs(log2_fc)
        
        # Find regions with effect size above threshold
        significant_bins = []
        
        for bin_idx in range(log2_fc.shape[0]):  # Iterate over genomic bins
            bin_effects = abs_log2_fc[bin_idx, :]  # Effects across all tracks of this type
            max_effect = np.max(bin_effects)
            mean_effect = np.mean(bin_effects)
            
            if max_effect > self.effect_threshold:
                significant_bins.append({
                    'bin_index': bin_idx,
                    'max_effect': max_effect,
                    'mean_effect': mean_effect,
                    'track_name': track_name,
                    'effect_direction': 'increase' if np.mean(log2_fc[bin_idx, :]) > 0 else 'decrease'
                })
                
        # Sort by effect size
        significant_bins.sort(key=lambda x: x['max_effect'], reverse=True)
        
        logger.info(f"Found {len(significant_bins)} significant bins for {track_name}")
        return significant_bins
        
    def find_candidate_enhancers(self, enformer_results):
        """Identify candidate enhancer regions from all tracks."""
        sequence_metadata = enformer_results['metadata']['sequence_metadata']
        
        all_significant_regions = []
        
        # Analyze each track type
        for track_name, track_data in enformer_results.items():
            if isinstance(track_data, dict) and 'log2_fold_change' in track_data:
                significant_regions = self.identify_significant_regions(track_data, track_name)
                all_significant_regions.extend(significant_regions)
                
        # Convert to DataFrame for easier analysis
        if all_significant_regions:
            regions_df = pd.DataFrame(all_significant_regions)
            
            # Add genomic coordinates
            bin_coords = self.calculate_genomic_coordinates(
                regions_df['bin_index'].values, sequence_metadata
            )
            
            coord_df = pd.DataFrame(bin_coords)
            regions_df = regions_df.merge(coord_df, on='bin_index')
            
        else:
            regions_df = pd.DataFrame()
            
        logger.info(f"Total significant regions identified: {len(regions_df)}")
        return regions_df
        
    def prioritize_candidate_enhancers(self, regions_df):
        """Prioritize candidate enhancers based on multiple criteria."""
        if len(regions_df) == 0:
            logger.warning("No significant regions found")
            return regions_df
            
        # Calculate composite score
        regions_df['composite_score'] = (
            regions_df['max_effect'] * 0.6 +  # Weight by effect size
            regions_df['mean_effect'] * 0.4   # Weight by consistency
        )
        
        # Group by track type and take top candidates
        top_candidates = []
        for track_name in regions_df['track_name'].unique():
            track_regions = regions_df[regions_df['track_name'] == track_name]
            top_track_regions = track_regions.nlargest(3, 'composite_score')
            top_candidates.append(top_track_regions)
            
        if top_candidates:
            prioritized_df = pd.concat(top_candidates, ignore_index=True)
            prioritized_df = prioritized_df.sort_values('composite_score', ascending=False)
        else:
            prioritized_df = regions_df
            
        return prioritized_df
        
    def visualize_track_differences(self, enformer_results, top_enhancers):
        """Create visualization of prediction differences along genomic coordinate."""
        sequence_metadata = enformer_results['metadata']['sequence_metadata']
        
        # Set up the plot
        fig, axes = plt.subplots(len(enformer_results) - 2, 1, figsize=(15, 4 * (len(enformer_results) - 2)))
        if len(enformer_results) - 2 == 1:
            axes = [axes]
            
        plot_idx = 0
        
        for track_name, track_data in enformer_results.items():
            if isinstance(track_data, dict) and 'log2_fold_change' in track_data:
                ax = axes[plot_idx]
                
                # Get log2 fold change data
                log2_fc = track_data['log2_fold_change']
                
                # Calculate mean across tracks (if multiple tracks per type)
                if log2_fc.ndim > 1:
                    mean_log2_fc = np.mean(log2_fc, axis=1)
                else:
                    mean_log2_fc = log2_fc
                    
                # Create genomic coordinates for x-axis
                bin_coords = self.calculate_genomic_coordinates(
                    range(len(mean_log2_fc)), sequence_metadata
                )
                
                x_coords = [coord['center'] for coord in bin_coords]
                
                # Plot the track
                ax.plot(x_coords, mean_log2_fc, linewidth=1, alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.axhline(y=self.effect_threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold: {self.effect_threshold}')
                ax.axhline(y=-self.effect_threshold, color='red', linestyle='--', alpha=0.5)
                
                # Highlight significant regions
                track_enhancers = top_enhancers[top_enhancers['track_name'] == track_name]
                for _, enhancer in track_enhancers.head(3).iterrows():  # Top 3 per track
                    ax.axvspan(enhancer['start'], enhancer['end'], alpha=0.3, color='orange')
                    ax.text(enhancer['center'], enhancer['max_effect'], 
                           f"Effect: {enhancer['max_effect']:.2f}", 
                           ha='center', fontsize=8)
                    
                # Mark SNP position
                snp_pos = sequence_metadata['position']
                ax.axvline(x=snp_pos, color='red', linestyle='-', linewidth=2, alpha=0.8, label='SNP position')
                
                ax.set_title(f'{track_name} - Log2 Fold Change', fontsize=12)
                ax.set_ylabel('Log2 Fold Change')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
                
        plt.xlabel(f'Genomic Position (Chr{sequence_metadata["chromosome"]})')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{self.target_snp}_enformer_track_differences.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Track differences plot saved to: {plot_file}")
        
        plt.show()
        return plot_file
        
    def save_enhancer_results(self, top_enhancers, sequence_metadata):
        """Save candidate enhancer results."""
        # Save detailed results
        enhancers_file = self.output_dir / f"{self.target_snp}_candidate_enhancers.csv"
        top_enhancers.to_csv(enhancers_file, index=False)
        logger.info(f"Candidate enhancers saved to: {enhancers_file}")
        
        # Create summary report
        if len(top_enhancers) > 0:
            top_enhancer = top_enhancers.iloc[0]
            
            summary = {
                'target_snp': self.target_snp,
                'disease': self.disease_name,
                'snp_chromosome': sequence_metadata['chromosome'],
                'snp_position': sequence_metadata['position'],
                'candidate_enhancer': {
                    'chromosome': int(top_enhancer['chr']),
                    'start': int(top_enhancer['start']),
                    'end': int(top_enhancer['end']),
                    'center': int(top_enhancer['center']),
                    'max_effect_size': float(top_enhancer['max_effect']),
                    'effect_direction': top_enhancer['effect_direction'],
                    'track_type': top_enhancer['track_name'],
                    'composite_score': float(top_enhancer['composite_score'])
                },
                'all_candidate_enhancers': len(top_enhancers),
                'analysis_parameters': {
                    'effect_threshold': self.effect_threshold
                }
            }
        else:
            summary = {
                'target_snp': self.target_snp,
                'disease': self.disease_name,
                'snp_chromosome': sequence_metadata['chromosome'],
                'snp_position': sequence_metadata['position'],
                'candidate_enhancer': None,
                'all_candidate_enhancers': 0,
                'analysis_parameters': {
                    'effect_threshold': self.effect_threshold
                }
            }
            
        summary_file = self.output_dir / f"{self.target_snp}_enhancer_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Enhancer analysis summary saved to: {summary_file}")
        
        return enhancers_file, summary_file, summary
        
    def identify_enhancers(self):
        """Run complete enhancer identification pipeline."""
        logger.info("Starting enhancer identification pipeline")
        
        # Load Enformer results
        enformer_results = self.load_enformer_results()
        
        # Find candidate enhancers
        candidate_regions = self.find_candidate_enhancers(enformer_results)
        
        # Prioritize candidates
        top_enhancers = self.prioritize_candidate_enhancers(candidate_regions)
        
        # Create visualizations
        plot_file = self.visualize_track_differences(enformer_results, top_enhancers)
        
        # Save results
        sequence_metadata = enformer_results['metadata']['sequence_metadata']
        enhancers_file, summary_file, summary = self.save_enhancer_results(top_enhancers, sequence_metadata)
        
        logger.info("Enhancer identification completed successfully")
        
        return {
            'target_snp': self.target_snp,
            'enhancers_file': enhancers_file,
            'summary_file': summary_file,
            'plot_file': plot_file,
            'n_candidates': len(top_enhancers),
            'top_enhancer': summary.get('candidate_enhancer')
        }


def main():
    parser = argparse.ArgumentParser(description='Identify functional enhancers from Enformer predictions')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--target-snp', help='Target SNP rsID')
    
    args = parser.parse_args()
    
    # Initialize identifier
    identifier = EnhancerIdentifier(args.config, args.target_snp)
    
    # Identify enhancers
    results = identifier.identify_enhancers()
    
    print("\n" + "="*50)
    print("ENHANCER IDENTIFICATION COMPLETED")
    print("="*50)
    print(f"Target SNP: {results['target_snp']}")
    print(f"Candidate enhancers found: {results['n_candidates']}")
    
    if results['top_enhancer']:
        top = results['top_enhancer']
        print(f"Top candidate enhancer:")
        print(f"  Location: chr{top['chromosome']}:{top['start']}-{top['end']}")
        print(f"  Effect size: {top['max_effect_size']:.3f}")
        print(f"  Direction: {top['effect_direction']}")
        print(f"  Track type: {top['track_type']}")
    else:
        print("No significant enhancers identified")
        
    print(f"Results saved to: {results['enhancers_file']}")
    print(f"Visualization: {results['plot_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
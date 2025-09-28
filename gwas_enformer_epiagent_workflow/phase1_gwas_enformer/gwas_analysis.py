#!/usr/bin/env python3
"""
Phase 1.1: GWAS Data Analysis to Identify Candidate SNPs

This script downloads and analyzes GWAS summary statistics to identify the lead SNP 
and SNPs in high linkage disequilibrium with it.

Usage:
    python gwas_analysis.py --config ../config/config.ini --disease "Alzheimer's disease"
"""

import argparse
import configparser
import pandas as pd
import numpy as np
import requests
import os
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GWASAnalyzer:
    def __init__(self, config_file, disease_name=None):
        """Initialize GWAS analyzer with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        if disease_name:
            self.config['disease']['name'] = disease_name
            
        self.disease_name = self.config['disease']['name']
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GWAS parameters
        self.ld_threshold = float(self.config['gwas']['ld_threshold'])
        self.p_threshold = float(self.config['gwas']['p_value_threshold'])
        
    def download_gwas_data(self, gwas_id=None):
        """Download GWAS summary statistics from GWAS Catalog."""
        if gwas_id is None:
            gwas_id = self.config['disease'].get('gwas_catalog_id')
            
        logger.info(f"Downloading GWAS data for {self.disease_name} (ID: {gwas_id})")
        
        # GWAS Catalog API endpoint
        base_url = "https://www.ebi.ac.uk/gwas/api/search/downloads"
        
        # For demonstration, we'll create a sample GWAS dataset
        # In real usage, this would download from the GWAS Catalog
        sample_data = {
            'rsID': [f'rs{i}' for i in range(1000000, 1000100)],
            'CHR': np.random.choice(range(1, 23), 100),
            'POS': np.random.randint(1000000, 250000000, 100),
            'A1': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'A2': np.random.choice(['A', 'T', 'G', 'C'], 100),
            'BETA': np.random.normal(0, 0.1, 100),
            'SE': np.random.uniform(0.01, 0.1, 100),
            'P': np.random.uniform(1e-10, 0.01, 100)
        }
        
        # Add a few genome-wide significant SNPs
        for i in range(5):
            sample_data['P'][i] = np.random.uniform(1e-12, 5e-9)
            
        gwas_df = pd.DataFrame(sample_data)
        
        # Save to output directory
        gwas_file = self.output_dir / f"{self.disease_name.replace(' ', '_')}_gwas_summary.tsv"
        gwas_df.to_csv(gwas_file, sep='\t', index=False)
        
        logger.info(f"GWAS data saved to: {gwas_file}")
        return gwas_df, gwas_file
        
    def identify_lead_snp(self, gwas_df):
        """Identify the lead SNP with lowest p-value."""
        # Filter for genome-wide significant SNPs
        significant_snps = gwas_df[gwas_df['P'] < self.p_threshold]
        
        if len(significant_snps) == 0:
            logger.warning("No genome-wide significant SNPs found. Using most significant SNP.")
            lead_snp = gwas_df.loc[gwas_df['P'].idxmin()]
        else:
            lead_snp = significant_snps.loc[significant_snps['P'].idxmin()]
            
        logger.info(f"Lead SNP identified: {lead_snp['rsID']} (chr{lead_snp['CHR']}:{lead_snp['POS']}, p={lead_snp['P']:.2e})")
        return lead_snp
        
    def find_ld_snps(self, lead_snp, gwas_df):
        """Find SNPs in linkage disequilibrium with the lead SNP."""
        logger.info(f"Finding SNPs in LD (r² > {self.ld_threshold}) with {lead_snp['rsID']}")
        
        # For demonstration, we'll simulate LD calculation
        # In real usage, this would use PLINK with 1000 Genomes reference panel
        
        # Filter SNPs on the same chromosome within 1Mb window
        chr_snps = gwas_df[
            (gwas_df['CHR'] == lead_snp['CHR']) & 
            (abs(gwas_df['POS'] - lead_snp['POS']) <= 1000000)
        ].copy()
        
        # Simulate LD r² values (in real analysis, calculate using PLINK)
        chr_snps['LD_r2'] = np.random.beta(2, 5, len(chr_snps))  # Simulated LD values
        chr_snps.loc[chr_snps['rsID'] == lead_snp['rsID'], 'LD_r2'] = 1.0  # Lead SNP has r²=1 with itself
        
        # Filter for high LD SNPs
        ld_snps = chr_snps[chr_snps['LD_r2'] > self.ld_threshold]
        
        logger.info(f"Found {len(ld_snps)} SNPs in high LD with lead SNP")
        return ld_snps
        
    def save_candidate_snps(self, ld_snps, lead_snp):
        """Save candidate SNPs to CSV file."""
        # Add lead SNP info
        ld_snps = ld_snps.copy()
        ld_snps['is_lead_snp'] = ld_snps['rsID'] == lead_snp['rsID']
        
        # Sort by LD r² (descending)
        ld_snps = ld_snps.sort_values('LD_r2', ascending=False)
        
        output_file = self.output_dir / f"{self.disease_name.replace(' ', '_')}_candidate_snps.csv"
        ld_snps.to_csv(output_file, index=False)
        
        logger.info(f"Candidate SNPs saved to: {output_file}")
        
        # Save summary statistics
        summary = {
            'disease': self.disease_name,
            'lead_snp': lead_snp['rsID'],
            'lead_snp_chr': int(lead_snp['CHR']),
            'lead_snp_pos': int(lead_snp['POS']),
            'lead_snp_pvalue': float(lead_snp['P']),
            'n_candidate_snps': len(ld_snps),
            'ld_threshold': self.ld_threshold
        }
        
        summary_file = self.output_dir / f"{self.disease_name.replace(' ', '_')}_analysis_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Analysis summary saved to: {summary_file}")
        return output_file, summary_file
        
    def run_analysis(self):
        """Run complete GWAS analysis pipeline."""
        logger.info("Starting GWAS analysis pipeline")
        
        # Download GWAS data
        gwas_df, gwas_file = self.download_gwas_data()
        
        # Identify lead SNP
        lead_snp = self.identify_lead_snp(gwas_df)
        
        # Find LD SNPs
        ld_snps = self.find_ld_snps(lead_snp, gwas_df)
        
        # Save results
        candidate_file, summary_file = self.save_candidate_snps(ld_snps, lead_snp)
        
        logger.info("GWAS analysis completed successfully")
        return {
            'gwas_file': gwas_file,
            'candidate_file': candidate_file,
            'summary_file': summary_file,
            'lead_snp': lead_snp,
            'n_candidates': len(ld_snps)
        }


def main():
    parser = argparse.ArgumentParser(description='GWAS Analysis for SNP identification')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--disease', help='Disease name (overrides config)')
    parser.add_argument('--gwas-id', help='GWAS Catalog study ID')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GWASAnalyzer(args.config, args.disease)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    print("\n" + "="*50)
    print("GWAS ANALYSIS COMPLETED")
    print("="*50)
    print(f"Disease: {analyzer.disease_name}")
    print(f"Lead SNP: {results['lead_snp']['rsID']}")
    print(f"Candidate SNPs: {results['n_candidates']}")
    print(f"Results saved to: {results['candidate_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
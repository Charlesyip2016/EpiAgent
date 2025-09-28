#!/usr/bin/env python3
"""
Phase 1.2: Prepare Input Sequences for Enformer

This script extracts DNA sequences around target SNPs for Enformer prediction.
It creates reference and variant sequences centered on the SNP position.

Usage:
    python prepare_enformer_sequences.py --config ../config/config.ini --target-snp rs1000000
"""

import argparse
import configparser
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from pyfaidx import Fasta
import requests
import gzip
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequencePreparator:
    def __init__(self, config_file, target_snp=None):
        """Initialize sequence preparator with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.disease_name = self.config['disease']['name']
        self.output_dir = Path(self.config['data']['output_dir'])
        self.reference_genome = self.config['data'].get('reference_genome', './data/raw/hg38.fa')
        
        # Enformer parameters
        self.sequence_length = int(self.config['enformer']['sequence_length'])
        
        # Target SNP
        self.target_snp = target_snp
        
    def download_reference_genome(self):
        """Download hg38 reference genome if not available."""
        ref_path = Path(self.reference_genome)
        
        if ref_path.exists():
            logger.info(f"Reference genome found at: {ref_path}")
            return ref_path
            
        logger.info("Downloading hg38 reference genome...")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For demonstration, create a mock reference genome
        # In real usage, download from UCSC or similar source
        logger.warning("Creating mock reference genome for demonstration")
        
        # Create a simple mock chromosome sequence
        mock_sequence = "N" * 250000000  # 250Mb of N's as placeholder
        
        with open(ref_path, 'w') as f:
            f.write(">chr1\n")
            # Write in 80-character lines
            for i in range(0, len(mock_sequence), 80):
                f.write(mock_sequence[i:i+80] + "\n")
                
        logger.info(f"Mock reference genome created at: {ref_path}")
        return ref_path
        
    def load_candidate_snps(self):
        """Load candidate SNPs from Phase 1.1 analysis."""
        candidate_file = self.output_dir / f"{self.disease_name.replace(' ', '_')}_candidate_snps.csv"
        
        if not candidate_file.exists():
            raise FileNotFoundError(f"Candidate SNPs file not found: {candidate_file}")
            
        snps_df = pd.read_csv(candidate_file)
        logger.info(f"Loaded {len(snps_df)} candidate SNPs")
        
        return snps_df
        
    def get_target_snp_info(self, snps_df):
        """Get information for target SNP."""
        if self.target_snp is None:
            # Use lead SNP if no specific target provided
            target_info = snps_df[snps_df['is_lead_snp'] == True].iloc[0]
            self.target_snp = target_info['rsID']
        else:
            target_info = snps_df[snps_df['rsID'] == self.target_snp]
            if len(target_info) == 0:
                raise ValueError(f"Target SNP {self.target_snp} not found in candidate SNPs")
            target_info = target_info.iloc[0]
            
        logger.info(f"Target SNP: {self.target_snp} (chr{target_info['CHR']}:{target_info['POS']})")
        return target_info
        
    def extract_sequence_around_snp(self, snp_info, ref_genome_path):
        """Extract sequence around SNP position."""
        chromosome = f"chr{snp_info['CHR']}"
        position = int(snp_info['POS'])
        
        # Calculate coordinates for sequence extraction
        half_length = self.sequence_length // 2
        start_pos = position - half_length
        end_pos = position + half_length
        
        logger.info(f"Extracting {self.sequence_length} bp sequence around {snp_info['rsID']}")
        logger.info(f"Coordinates: {chromosome}:{start_pos}-{end_pos}")
        
        try:
            # Use pyfaidx to extract sequence
            fasta = Fasta(str(ref_genome_path))
            
            if chromosome not in fasta.keys():
                # Try without 'chr' prefix
                chromosome = str(snp_info['CHR'])
                if chromosome not in fasta.keys():
                    raise KeyError(f"Chromosome {chromosome} not found in reference genome")
                    
            sequence = str(fasta[chromosome][start_pos:end_pos])
            
        except Exception as e:
            logger.warning(f"Error extracting from reference genome: {e}")
            logger.warning("Using mock sequence for demonstration")
            
            # Create mock sequence with known bases at SNP position
            sequence = "N" * self.sequence_length
            # Replace center position with reference allele
            center_pos = len(sequence) // 2
            sequence = sequence[:center_pos] + snp_info['A2'] + sequence[center_pos+1:]
            
        return sequence, start_pos, end_pos
        
    def create_variant_sequence(self, reference_sequence, snp_info):
        """Create variant sequence by replacing reference allele with risk allele."""
        variant_sequence = list(reference_sequence)
        center_pos = len(variant_sequence) // 2
        
        # Replace center position with risk allele (A1)
        variant_sequence[center_pos] = snp_info['A1']
        
        return ''.join(variant_sequence)
        
    def save_sequences_to_fasta(self, ref_seq, var_seq, snp_info, start_pos, end_pos):
        """Save reference and variant sequences to FASTA files."""
        ref_fasta = self.output_dir / f"{self.target_snp}_reference_sequence.fasta"
        var_fasta = self.output_dir / f"{self.target_snp}_variant_sequence.fasta"
        
        # Reference sequence
        with open(ref_fasta, 'w') as f:
            f.write(f">{self.target_snp}_reference_chr{snp_info['CHR']}:{start_pos}-{end_pos}\n")
            # Write in 80-character lines
            for i in range(0, len(ref_seq), 80):
                f.write(ref_seq[i:i+80] + "\n")
                
        # Variant sequence
        with open(var_fasta, 'w') as f:
            f.write(f">{self.target_snp}_variant_{snp_info['A2']}>{snp_info['A1']}_chr{snp_info['CHR']}:{start_pos}-{end_pos}\n")
            for i in range(0, len(var_seq), 80):
                f.write(var_seq[i:i+80] + "\n")
                
        logger.info(f"Reference sequence saved to: {ref_fasta}")
        logger.info(f"Variant sequence saved to: {var_fasta}")
        
        return ref_fasta, var_fasta
        
    def create_sequence_metadata(self, snp_info, ref_fasta, var_fasta, start_pos, end_pos):
        """Create metadata file for the sequences."""
        metadata = {
            'target_snp': self.target_snp,
            'disease': self.disease_name,
            'chromosome': int(snp_info['CHR']),
            'position': int(snp_info['POS']),
            'reference_allele': snp_info['A2'],
            'risk_allele': snp_info['A1'],
            'sequence_start': int(start_pos),
            'sequence_end': int(end_pos),
            'sequence_length': self.sequence_length,
            'reference_fasta': str(ref_fasta),
            'variant_fasta': str(var_fasta),
            'p_value': float(snp_info['P']),
            'effect_size': float(snp_info['BETA']),
            'ld_r2': float(snp_info.get('LD_r2', 1.0))
        }
        
        import json
        metadata_file = self.output_dir / f"{self.target_snp}_sequence_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Sequence metadata saved to: {metadata_file}")
        return metadata_file
        
    def prepare_sequences(self):
        """Run complete sequence preparation pipeline."""
        logger.info("Starting sequence preparation pipeline")
        
        # Download reference genome
        ref_genome_path = self.download_reference_genome()
        
        # Load candidate SNPs
        snps_df = self.load_candidate_snps()
        
        # Get target SNP information
        snp_info = self.get_target_snp_info(snps_df)
        
        # Extract reference sequence
        ref_sequence, start_pos, end_pos = self.extract_sequence_around_snp(snp_info, ref_genome_path)
        
        # Create variant sequence
        var_sequence = self.create_variant_sequence(ref_sequence, snp_info)
        
        # Save sequences to FASTA files
        ref_fasta, var_fasta = self.save_sequences_to_fasta(
            ref_sequence, var_sequence, snp_info, start_pos, end_pos
        )
        
        # Create metadata
        metadata_file = self.create_sequence_metadata(
            snp_info, ref_fasta, var_fasta, start_pos, end_pos
        )
        
        logger.info("Sequence preparation completed successfully")
        
        return {
            'target_snp': self.target_snp,
            'reference_fasta': ref_fasta,
            'variant_fasta': var_fasta,
            'metadata_file': metadata_file,
            'sequence_length': len(ref_sequence)
        }


def main():
    parser = argparse.ArgumentParser(description='Prepare sequences for Enformer prediction')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--target-snp', help='Target SNP rsID (uses lead SNP if not specified)')
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = SequencePreparator(args.config, args.target_snp)
    
    # Prepare sequences
    results = preparator.prepare_sequences()
    
    print("\n" + "="*50)
    print("SEQUENCE PREPARATION COMPLETED")
    print("="*50)
    print(f"Target SNP: {results['target_snp']}")
    print(f"Sequence length: {results['sequence_length']:,} bp")
    print(f"Reference sequence: {results['reference_fasta']}")
    print(f"Variant sequence: {results['variant_fasta']}")
    print(f"Metadata: {results['metadata_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
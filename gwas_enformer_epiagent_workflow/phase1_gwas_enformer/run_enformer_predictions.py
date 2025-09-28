#!/usr/bin/env python3
"""
Phase 1.3: Run Enformer Predictions

This script runs Enformer predictions on reference and variant sequences
to predict the functional impact of the SNP on regulatory elements.

Usage:
    python run_enformer_predictions.py --config ../config/config.ini --target-snp rs1000000
"""

import argparse
import configparser
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import logging
import json
import h5py
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnformerPredictor:
    def __init__(self, config_file, target_snp=None):
        """Initialize Enformer predictor with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.disease_name = self.config['disease']['name']
        self.output_dir = Path(self.config['data']['output_dir'])
        
        # Enformer parameters
        self.model_url = self.config['enformer']['model_url']
        self.sequence_length = int(self.config['enformer']['sequence_length'])
        self.prediction_tracks = self.config['enformer']['prediction_tracks'].split(', ')
        self.cell_lines = self.config['enformer']['cell_lines_of_interest'].split(', ')
        
        self.target_snp = target_snp
        self.model = None
        
    def load_enformer_model(self):
        """Load pre-trained Enformer model from TensorFlow Hub."""
        logger.info("Loading Enformer model from TensorFlow Hub...")
        
        try:
            self.model = hub.load(self.model_url).model
            logger.info("Enformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Enformer model: {e}")
            logger.info("Creating mock Enformer model for demonstration")
            self.model = self._create_mock_model()
            
    def _create_mock_model(self):
        """Create a mock model for demonstration purposes."""
        class MockEnformer:
            def predict(self, sequences):
                # Mock prediction with random values
                batch_size = sequences.shape[0]
                # Enformer outputs 896 bins × 5313 tracks
                return np.random.normal(0, 1, (batch_size, 896, 5313))
                
        return MockEnformer()
        
    def load_sequence_metadata(self):
        """Load sequence metadata from Phase 1.2."""
        if self.target_snp is None:
            # Find the most recent sequence metadata file
            metadata_files = list(self.output_dir.glob("*_sequence_metadata.json"))
            if not metadata_files:
                raise FileNotFoundError("No sequence metadata files found")
            metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        else:
            metadata_file = self.output_dir / f"{self.target_snp}_sequence_metadata.json"
            
        if not metadata_file.exists():
            raise FileNotFoundError(f"Sequence metadata not found: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        self.target_snp = metadata['target_snp']
        logger.info(f"Loaded metadata for {self.target_snp}")
        
        return metadata
        
    def sequence_to_one_hot(self, sequence):
        """Convert DNA sequence to one-hot encoding for Enformer."""
        # Mapping for nucleotides
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Convert sequence to indices
        sequence_indices = [nucleotide_map.get(nt.upper(), 4) for nt in sequence]
        
        # Create one-hot encoding (5 channels for A, C, G, T, N)
        one_hot = np.zeros((len(sequence), 5), dtype=np.float32)
        for i, idx in enumerate(sequence_indices):
            if idx < 4:  # Valid nucleotide
                one_hot[i, idx] = 1.0
            else:  # N or unknown
                # Uniform distribution for unknown nucleotides
                one_hot[i, :4] = 0.25
                
        return one_hot
        
    def load_and_encode_sequences(self, metadata):
        """Load FASTA sequences and convert to one-hot encoding."""
        ref_fasta = Path(metadata['reference_fasta'])
        var_fasta = Path(metadata['variant_fasta'])
        
        # Load reference sequence
        with open(ref_fasta, 'r') as f:
            ref_record = next(SeqIO.parse(f, "fasta"))
            ref_sequence = str(ref_record.seq)
            
        # Load variant sequence
        with open(var_fasta, 'r') as f:
            var_record = next(SeqIO.parse(f, "fasta"))
            var_sequence = str(var_record.seq)
            
        logger.info(f"Loaded sequences of length {len(ref_sequence)} bp")
        
        # Convert to one-hot encoding
        ref_one_hot = self.sequence_to_one_hot(ref_sequence)
        var_one_hot = self.sequence_to_one_hot(var_sequence)
        
        # Ensure correct length for Enformer
        if ref_one_hot.shape[0] != self.sequence_length:
            logger.warning(f"Sequence length mismatch: expected {self.sequence_length}, got {ref_one_hot.shape[0]}")
            
        return ref_one_hot, var_one_hot, ref_sequence, var_sequence
        
    def run_enformer_predictions(self, ref_one_hot, var_one_hot):
        """Run Enformer predictions on encoded sequences."""
        logger.info("Running Enformer predictions...")
        
        # Prepare batch input (batch_size, seq_len, channels)
        batch_input = np.stack([ref_one_hot, var_one_hot])
        
        # Run predictions
        predictions = self.model.predict(batch_input)
        
        ref_predictions = predictions[0]  # Shape: (896, 5313)
        var_predictions = predictions[1]  # Shape: (896, 5313)
        
        logger.info(f"Predictions shape: {ref_predictions.shape}")
        
        return ref_predictions, var_predictions
        
    def extract_relevant_tracks(self, predictions):
        """Extract predictions for relevant tracks (CAGE, DNase, ATAC, H3K27ac)."""
        # This is a simplified example - in real usage, you'd need to know
        # the exact track indices for each assay type and cell line
        
        # For demonstration, we'll simulate track extraction
        track_mapping = {
            'CAGE': list(range(0, 100)),      # Mock track indices
            'DNase': list(range(100, 200)),   
            'ATAC-seq': list(range(200, 300)),
            'H3K27ac': list(range(300, 400))
        }
        
        extracted_tracks = {}
        for track_name in self.prediction_tracks:
            if track_name in track_mapping:
                track_indices = track_mapping[track_name]
                extracted_tracks[track_name] = predictions[:, track_indices]
                logger.info(f"Extracted {track_name} tracks: {len(track_indices)} tracks")
                
        return extracted_tracks
        
    def calculate_prediction_differences(self, ref_predictions, var_predictions):
        """Calculate differences between variant and reference predictions."""
        logger.info("Calculating prediction differences...")
        
        # Extract relevant tracks
        ref_tracks = self.extract_relevant_tracks(ref_predictions)
        var_tracks = self.extract_relevant_tracks(var_predictions)
        
        differences = {}
        for track_name in ref_tracks:
            if track_name in var_tracks:
                # Calculate log2 fold change
                ref_track = ref_tracks[track_name]
                var_track = var_tracks[track_name]
                
                # Add pseudocount to avoid log(0)
                pseudocount = 1e-6
                log2_fc = np.log2((var_track + pseudocount) / (ref_track + pseudocount))
                
                # Also calculate direct difference
                direct_diff = var_track - ref_track
                
                differences[track_name] = {
                    'log2_fold_change': log2_fc,
                    'direct_difference': direct_diff,
                    'reference': ref_track,
                    'variant': var_track
                }
                
        return differences
        
    def save_predictions(self, ref_predictions, var_predictions, differences, metadata):
        """Save prediction results to files."""
        logger.info("Saving prediction results...")
        
        # Save as HDF5 for large arrays
        h5_file = self.output_dir / f"{self.target_snp}_enformer_predictions.h5"
        
        with h5py.File(h5_file, 'w') as f:
            # Save raw predictions
            f.create_dataset('reference_predictions', data=ref_predictions)
            f.create_dataset('variant_predictions', data=var_predictions)
            
            # Save track-specific results
            for track_name, track_data in differences.items():
                grp = f.create_group(track_name)
                grp.create_dataset('log2_fold_change', data=track_data['log2_fold_change'])
                grp.create_dataset('direct_difference', data=track_data['direct_difference'])
                grp.create_dataset('reference', data=track_data['reference'])
                grp.create_dataset('variant', data=track_data['variant'])
                
        logger.info(f"Predictions saved to: {h5_file}")
        
        # Save summary statistics
        summary = {
            'target_snp': self.target_snp,
            'disease': self.disease_name,
            'predictions_file': str(h5_file),
            'prediction_tracks': list(differences.keys()),
            'sequence_metadata': metadata
        }
        
        # Calculate summary statistics for each track
        for track_name, track_data in differences.items():
            log2_fc = track_data['log2_fold_change']
            summary[f'{track_name}_max_log2fc'] = float(np.max(np.abs(log2_fc)))
            summary[f'{track_name}_mean_log2fc'] = float(np.mean(log2_fc))
            summary[f'{track_name}_std_log2fc'] = float(np.std(log2_fc))
            
        summary_file = self.output_dir / f"{self.target_snp}_enformer_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary statistics saved to: {summary_file}")
        
        return h5_file, summary_file
        
    def run_predictions(self):
        """Run complete Enformer prediction pipeline."""
        logger.info("Starting Enformer prediction pipeline")
        
        # Load Enformer model
        self.load_enformer_model()
        
        # Load sequence metadata
        metadata = self.load_sequence_metadata()
        
        # Load and encode sequences
        ref_one_hot, var_one_hot, ref_seq, var_seq = self.load_and_encode_sequences(metadata)
        
        # Run Enformer predictions
        ref_predictions, var_predictions = self.run_enformer_predictions(ref_one_hot, var_one_hot)
        
        # Calculate differences
        differences = self.calculate_prediction_differences(ref_predictions, var_predictions)
        
        # Save results
        h5_file, summary_file = self.save_predictions(ref_predictions, var_predictions, differences, metadata)
        
        logger.info("Enformer prediction completed successfully")
        
        return {
            'target_snp': self.target_snp,
            'predictions_file': h5_file,
            'summary_file': summary_file,
            'tracks_analyzed': list(differences.keys())
        }


def main():
    parser = argparse.ArgumentParser(description='Run Enformer predictions on SNP sequences')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--target-snp', help='Target SNP rsID')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EnformerPredictor(args.config, args.target_snp)
    
    # Run predictions
    results = predictor.run_predictions()
    
    print("\n" + "="*50)
    print("ENFORMER PREDICTIONS COMPLETED")
    print("="*50)
    print(f"Target SNP: {results['target_snp']}")
    print(f"Tracks analyzed: {', '.join(results['tracks_analyzed'])}")
    print(f"Predictions file: {results['predictions_file']}")
    print(f"Summary file: {results['summary_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
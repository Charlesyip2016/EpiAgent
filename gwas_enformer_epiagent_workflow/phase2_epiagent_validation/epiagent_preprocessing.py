#!/usr/bin/env python3
"""
Phase 2.1: EpiAgent Data Preprocessing

This script preprocesses scATAC-seq data following EpiAgent methodology,
including TF-IDF transformation and cell sentence generation.

Usage:
    python epiagent_preprocessing.py --config ../config/config.ini --dataset-id GSE12345
"""

import argparse
import configparser
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
import logging
import json
import requests
import gzip
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpiAgentPreprocessor:
    def __init__(self, config_file, dataset_id=None):
        """Initialize EpiAgent preprocessor with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.disease_name = self.config['disease']['name']
        self.target_tissue = self.config['disease']['target_tissue']
        self.output_dir = Path(self.config['data']['output_dir'])
        self.processed_dir = Path(self.config['data']['processed_data_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # EpiAgent parameters
        self.ccre_reference_file = self.config['epiagent'].get('ccre_reference_file', './data/ccre_coordinates_hg38.bed')
        self.max_seq_length = int(self.config['epiagent']['max_sequence_length'])
        
        self.dataset_id = dataset_id or self.config['data']['scatac_dataset_id']
        
    def download_scatac_dataset(self):
        """Download and load scATAC-seq dataset."""
        logger.info(f"Loading scATAC-seq dataset: {self.dataset_id}")
        
        # For demonstration, create a mock scATAC-seq dataset
        # In real usage, this would download from GEO/SRA or load from files
        logger.warning("Creating mock scATAC-seq dataset for demonstration")
        
        # Create mock data
        n_cells = 5000
        n_features = 100000  # Mock number of peaks/cCREs
        
        # Create sparse matrix (most entries are 0 in scATAC-seq)
        data = sp.random(n_cells, n_features, density=0.01, format='csr')
        data.data = np.ones_like(data.data)  # Binary accessibility
        
        # Create cell metadata
        cell_types = np.random.choice(['microglia', 'astrocytes', 'neurons', 'oligodendrocytes'], n_cells)
        batch_ids = np.random.choice(['batch1', 'batch2', 'batch3'], n_cells)
        
        obs = pd.DataFrame({
            'cell_type': cell_types,
            'batch': batch_ids,
            'n_genes': np.random.poisson(3000, n_cells),
            'tissue': self.target_tissue
        })
        obs.index = [f'cell_{i}' for i in range(n_cells)]
        
        # Create feature metadata (peaks/cCREs)
        var = pd.DataFrame({
            'chr': np.random.choice(range(1, 23), n_features),
            'start': np.random.randint(1000000, 200000000, n_features),
            'end': np.random.randint(1000000, 200000000, n_features)
        })
        var['feature_name'] = [f'peak_{i}' for i in range(n_features)]
        var.index = var['feature_name']
        
        # Create AnnData object
        adata = ad.AnnData(X=data, obs=obs, var=var)
        
        # Save raw dataset
        raw_file = self.processed_dir / f"{self.dataset_id}_raw.h5ad"
        adata.write(raw_file)
        logger.info(f"Raw dataset saved to: {raw_file}")
        
        return adata
        
    def load_ccre_reference(self):
        """Load cCRE reference coordinates."""
        ccre_file = Path(self.ccre_reference_file)
        
        if ccre_file.exists():
            logger.info(f"Loading cCRE reference from: {ccre_file}")
            ccre_df = pd.read_csv(ccre_file, sep='\t', header=None,
                                names=['chr', 'start', 'end', 'ccre_id'])
        else:
            logger.warning("cCRE reference file not found. Creating mock reference.")
            # Create mock cCRE reference
            n_ccres = 1355445  # EpiAgent standard
            ccre_df = pd.DataFrame({
                'chr': [f'chr{i}' for i in np.random.choice(range(1, 23), n_ccres)],
                'start': np.random.randint(1000, 250000000, n_ccres),
                'end': np.random.randint(1000, 250000000, n_ccres),
                'ccre_id': [f'cCRE_{i}' for i in range(n_ccres)]
            })
            # Ensure end > start
            ccre_df['end'] = ccre_df['start'] + np.random.randint(200, 2000, n_ccres)
            
            # Save mock reference
            ccre_file.parent.mkdir(parents=True, exist_ok=True)
            ccre_df.to_csv(ccre_file, sep='\t', header=False, index=False)
            
        logger.info(f"Loaded {len(ccre_df)} cCREs")
        return ccre_df
        
    def map_peaks_to_ccres(self, adata, ccre_df):
        """Map scATAC-seq peaks to unified cCRE reference set."""
        logger.info("Mapping peaks to cCRE reference set...")
        
        # For demonstration, create a mock mapping
        # In real usage, this would use bedtools intersect or similar
        
        n_peaks = adata.n_vars
        n_ccres = len(ccre_df)
        
        # Create random mapping (in reality, this would be based on genomic overlap)
        peak_to_ccre_mapping = {}
        mapped_peaks = 0
        
        for i in range(n_peaks):
            # Some peaks map to cCREs, some don't
            if np.random.random() < 0.7:  # 70% mapping rate
                ccre_idx = np.random.randint(0, n_ccres)
                peak_to_ccre_mapping[i] = ccre_idx
                mapped_peaks += 1
                
        logger.info(f"Mapped {mapped_peaks}/{n_peaks} peaks to cCREs")
        
        # Create cell-by-cCRE matrix
        cell_ccre_matrix = sp.lil_matrix((adata.n_obs, n_ccres))
        
        for peak_idx, ccre_idx in peak_to_ccre_mapping.items():
            # Add peak accessibility to corresponding cCRE
            cell_ccre_matrix[:, ccre_idx] += adata.X[:, peak_idx]
            
        # Convert to CSR format for efficiency
        cell_ccre_matrix = cell_ccre_matrix.tocsr()
        
        # Binarize (any accessibility = 1)
        cell_ccre_matrix.data = np.ones_like(cell_ccre_matrix.data)
        
        # Create new AnnData with cCRE features
        ccre_var = pd.DataFrame({
            'ccre_id': ccre_df['ccre_id'],
            'chr': ccre_df['chr'],
            'start': ccre_df['start'],
            'end': ccre_df['end']
        })
        ccre_var.index = ccre_var['ccre_id']
        
        adata_ccre = ad.AnnData(X=cell_ccre_matrix, obs=adata.obs.copy(), var=ccre_var)
        
        logger.info(f"Created cell-by-cCRE matrix: {adata_ccre.shape}")
        return adata_ccre
        
    def apply_tfidf_transformation(self, adata_ccre):
        """Apply TF-IDF transformation to cell-by-cCRE matrix."""
        logger.info("Applying TF-IDF transformation...")
        
        # TF-IDF transformation following EpiAgent methodology
        tfidf = TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=False)
        
        # Fit and transform the data
        X_tfidf = tfidf.fit_transform(adata_ccre.X)
        
        # Create new AnnData object with TF-IDF values
        adata_tfidf = ad.AnnData(X=X_tfidf, obs=adata_ccre.obs.copy(), var=adata_ccre.var.copy())
        
        # Store TF-IDF parameters
        adata_tfidf.uns['tfidf_params'] = {
            'idf_': tfidf.idf_.tolist(),
            'norm': tfidf.norm,
            'use_idf': tfidf.use_idf,
            'sublinear_tf': tfidf.sublinear_tf
        }
        
        logger.info("TF-IDF transformation completed")
        return adata_tfidf
        
    def generate_cell_sentences(self, adata_tfidf):
        """Generate cell sentences by ranking accessible cCREs by TF-IDF values."""
        logger.info("Generating cell sentences...")
        
        cell_sentences = []
        
        for cell_idx in range(adata_tfidf.n_obs):
            # Get TF-IDF values for this cell
            cell_tfidf = adata_tfidf.X[cell_idx].toarray().flatten()
            
            # Find non-zero (accessible) cCREs
            accessible_ccres = np.where(cell_tfidf > 0)[0]
            
            if len(accessible_ccres) > 0:
                # Get TF-IDF scores for accessible cCREs
                accessible_scores = cell_tfidf[accessible_ccres]
                
                # Sort by TF-IDF score (descending)
                sorted_indices = np.argsort(accessible_scores)[::-1]
                sorted_ccres = accessible_ccres[sorted_indices]
                
                # Limit to max sequence length (reserve space for [CLS] and [SEP])
                max_ccres = self.max_seq_length - 2
                if len(sorted_ccres) > max_ccres:
                    sorted_ccres = sorted_ccres[:max_ccres]
                
                # Convert to list of cCRE indices (add offset if needed)
                cell_sentence = [ccre_idx + 4 for ccre_idx in sorted_ccres]  # +4 offset for special tokens
            else:
                cell_sentence = []
                
            cell_sentences.append(cell_sentence)
            
        logger.info(f"Generated cell sentences for {len(cell_sentences)} cells")
        
        # Add cell sentences to AnnData
        adata_tfidf.obs['cell_sentences'] = [str(sentence) for sentence in cell_sentences]
        
        return adata_tfidf, cell_sentences
        
    def save_processed_data(self, adata_processed, cell_sentences):
        """Save processed data for EpiAgent input."""
        # Save processed AnnData
        processed_file = self.processed_dir / f"{self.dataset_id}_processed_epiagent.h5ad"
        adata_processed.write(processed_file)
        logger.info(f"Processed data saved to: {processed_file}")
        
        # Save cell sentences separately for easier loading
        sentences_file = self.processed_dir / f"{self.dataset_id}_cell_sentences.json"
        with open(sentences_file, 'w') as f:
            json.dump(cell_sentences, f)
        logger.info(f"Cell sentences saved to: {sentences_file}")
        
        # Create preprocessing summary
        summary = {
            'dataset_id': self.dataset_id,
            'disease': self.disease_name,
            'target_tissue': self.target_tissue,
            'n_cells': adata_processed.n_obs,
            'n_ccres': adata_processed.n_vars,
            'avg_sentence_length': np.mean([len(sentence) for sentence in cell_sentences]),
            'median_sentence_length': np.median([len(sentence) for sentence in cell_sentences]),
            'processed_file': str(processed_file),
            'cell_sentences_file': str(sentences_file),
            'preprocessing_parameters': {
                'max_sequence_length': self.max_seq_length,
                'tfidf_applied': True
            }
        }
        
        summary_file = self.processed_dir / f"{self.dataset_id}_preprocessing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Preprocessing summary saved to: {summary_file}")
        
        return processed_file, sentences_file, summary_file
        
    def preprocess_data(self):
        """Run complete EpiAgent preprocessing pipeline."""
        logger.info("Starting EpiAgent preprocessing pipeline")
        
        # Download/load scATAC-seq dataset
        adata = self.download_scatac_dataset()
        
        # Load cCRE reference
        ccre_df = self.load_ccre_reference()
        
        # Map peaks to cCREs
        adata_ccre = self.map_peaks_to_ccres(adata, ccre_df)
        
        # Apply TF-IDF transformation
        adata_tfidf = self.apply_tfidf_transformation(adata_ccre)
        
        # Generate cell sentences
        adata_processed, cell_sentences = self.generate_cell_sentences(adata_tfidf)
        
        # Save processed data
        processed_file, sentences_file, summary_file = self.save_processed_data(adata_processed, cell_sentences)
        
        logger.info("EpiAgent preprocessing completed successfully")
        
        return {
            'dataset_id': self.dataset_id,
            'processed_file': processed_file,
            'sentences_file': sentences_file,
            'summary_file': summary_file,
            'n_cells': adata_processed.n_obs,
            'n_ccres': adata_processed.n_vars,
            'avg_sentence_length': np.mean([len(sentence) for sentence in cell_sentences])
        }


def main():
    parser = argparse.ArgumentParser(description='Preprocess scATAC-seq data for EpiAgent')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--dataset-id', help='Dataset identifier (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = EpiAgentPreprocessor(args.config, args.dataset_id)
    
    # Run preprocessing
    results = preprocessor.preprocess_data()
    
    print("\n" + "="*50)
    print("EPIAGENT PREPROCESSING COMPLETED")
    print("="*50)
    print(f"Dataset: {results['dataset_id']}")
    print(f"Cells processed: {results['n_cells']:,}")
    print(f"cCREs: {results['n_ccres']:,}")
    print(f"Average sentence length: {results['avg_sentence_length']:.1f}")
    print(f"Processed data: {results['processed_file']}")
    print(f"Cell sentences: {results['sentences_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
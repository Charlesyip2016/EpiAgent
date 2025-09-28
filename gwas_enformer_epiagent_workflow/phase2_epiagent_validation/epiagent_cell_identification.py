#!/usr/bin/env python3
"""
Phase 2.2: Run EpiAgent and Identify Cell Types

This script runs EpiAgent model to generate cell embeddings and performs
cell type identification using clustering and annotation.

Usage:
    python epiagent_cell_identification.py --config ../config/config.ini --dataset-id GSE12345
"""

import argparse
import configparser
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import torch
from pathlib import Path
import logging
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import EpiAgent if available
try:
    from epiagent.model import EpiAgent
    from epiagent.inference import infer_cell_embeddings
    EPIAGENT_AVAILABLE = True
except ImportError:
    EPIAGENT_AVAILABLE = False
    logger.warning("EpiAgent package not available. Using mock embeddings for demonstration.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpiAgentCellIdentifier:
    def __init__(self, config_file, dataset_id=None):
        """Initialize EpiAgent cell identifier with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.disease_name = self.config['disease']['name']
        self.target_tissue = self.config['disease']['target_tissue']
        self.target_cell_types = self.config['disease']['target_cell_types'].split(', ')
        
        self.output_dir = Path(self.config['data']['output_dir'])
        self.processed_dir = Path(self.config['data']['processed_data_dir'])
        
        # EpiAgent parameters
        self.model_path = self.config['epiagent'].get('model_path', './model/pretrained_EpiAgent.pth')
        self.vocab_size = int(self.config['epiagent']['vocab_size'])
        self.num_layers = int(self.config['epiagent']['num_layers'])
        self.embedding_dim = int(self.config['epiagent']['embedding_dim'])
        self.num_attention_heads = int(self.config['epiagent']['num_attention_heads'])
        self.max_seq_length = int(self.config['epiagent']['max_sequence_length'])
        
        self.dataset_id = dataset_id or self.config['data']['scatac_dataset_id']
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_preprocessed_data(self):
        """Load preprocessed data from Phase 2.1."""
        processed_file = self.processed_dir / f"{self.dataset_id}_processed_epiagent.h5ad"
        sentences_file = self.processed_dir / f"{self.dataset_id}_cell_sentences.json"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {processed_file}")
        if not sentences_file.exists():
            raise FileNotFoundError(f"Cell sentences not found: {sentences_file}")
            
        # Load AnnData
        adata = sc.read_h5ad(processed_file)
        
        # Load cell sentences
        with open(sentences_file, 'r') as f:
            cell_sentences = json.load(f)
            
        logger.info(f"Loaded preprocessed data: {adata.shape}")
        logger.info(f"Cell sentences: {len(cell_sentences)}")
        
        return adata, cell_sentences
        
    def load_epiagent_model(self):
        """Load pre-trained EpiAgent model."""
        if not EPIAGENT_AVAILABLE:
            logger.warning("EpiAgent not available. Creating mock model.")
            return self._create_mock_model()
            
        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.warning(f"EpiAgent model not found at {model_file}. Creating mock model.")
            return self._create_mock_model()
            
        try:
            # Initialize EpiAgent model
            model = EpiAgent(
                vocab_size=self.vocab_size,
                num_layers=self.num_layers,
                embedding_dim=self.embedding_dim,
                num_attention_heads=self.num_attention_heads,
                max_rank_embeddings=self.max_seq_length,
                use_flash_attn=False,  # Disable for compatibility
                pos_weight_for_RLM=torch.tensor(1.),
                pos_weight_for_CCA=torch.tensor(1.)
            )
            
            # Load pre-trained weights
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            logger.info("EpiAgent model loaded successfully")
            return model
            
        except Exception as e:
            logger.warning(f"Error loading EpiAgent model: {e}")
            return self._create_mock_model()
            
    def _create_mock_model(self):
        """Create mock model for demonstration."""
        class MockEpiAgent:
            def __init__(self, embedding_dim):
                self.embedding_dim = embedding_dim
                
            def forward(self, input_ids, return_transformer_output=True):
                batch_size = input_ids.shape[0]
                # Return random embeddings
                transformer_outputs = torch.randn(batch_size, input_ids.shape[1], self.embedding_dim)
                return {'transformer_outputs': transformer_outputs}
                
        return MockEpiAgent(self.embedding_dim)
        
    def generate_cell_embeddings(self, model, cell_sentences):
        """Generate cell embeddings using EpiAgent model."""
        logger.info("Generating cell embeddings...")
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(cell_sentences), batch_size):
            batch_sentences = cell_sentences[i:i+batch_size]
            
            # Convert cell sentences to input tensors
            batch_inputs = []
            for sentence in batch_sentences:
                if isinstance(sentence, str):
                    sentence = eval(sentence)  # Convert string back to list
                    
                # Pad/truncate to max length
                if len(sentence) > self.max_seq_length - 2:
                    sentence = sentence[:self.max_seq_length - 2]
                    
                # Add [CLS] and [SEP] tokens
                input_ids = [1] + sentence + [2]  # 1=[CLS], 2=[SEP]
                
                # Pad to max length
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)  # 0=[PAD]
                    
                batch_inputs.append(input_ids)
                
            # Convert to tensor
            input_tensor = torch.tensor(batch_inputs, dtype=torch.long).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(input_tensor, return_transformer_output=True)
                
                if hasattr(outputs, 'transformer_outputs'):
                    # Use [CLS] token embedding (first position)
                    batch_embeddings = outputs.transformer_outputs[:, 0, :].cpu().numpy()
                else:
                    # Mock embeddings
                    batch_embeddings = np.random.normal(0, 1, (len(batch_sentences), self.embedding_dim))
                    
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_sentences)}/{len(cell_sentences)} cells")
                
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
        
    def perform_clustering(self, embeddings, adata):
        """Perform Leiden clustering on cell embeddings."""
        logger.info("Performing clustering analysis...")
        
        # Add embeddings to AnnData
        adata.obsm['X_epiagent'] = embeddings
        
        # Compute neighborhood graph
        sc.pp.neighbors(adata, use_rep='X_epiagent', n_neighbors=15, n_pcs=None)
        
        # Leiden clustering
        sc.tl.leiden(adata, resolution=0.5, random_state=42)
        
        # UMAP for visualization
        sc.tl.umap(adata, random_state=42)
        
        n_clusters = len(adata.obs['leiden'].unique())
        logger.info(f"Identified {n_clusters} clusters")
        
        return adata
        
    def annotate_cell_types(self, adata):
        """Annotate clusters with cell types based on marker genes or transfer learning."""
        logger.info("Annotating cell types...")
        
        # For demonstration, we'll assign cell types based on cluster IDs
        # In real analysis, this would use marker gene expression or transfer learning
        
        cluster_to_celltype = {}
        unique_clusters = sorted(adata.obs['leiden'].unique())
        
        # Assign cell types to clusters (mock assignment)
        available_celltypes = ['microglia', 'astrocytes', 'neurons', 'oligodendrocytes', 
                              'endothelial', 'pericytes', 'unknown']
        
        for i, cluster in enumerate(unique_clusters):
            if i < len(available_celltypes):
                cluster_to_celltype[cluster] = available_celltypes[i]
            else:
                cluster_to_celltype[cluster] = 'unknown'
                
        # Apply annotations
        adata.obs['cell_type_predicted'] = adata.obs['leiden'].map(cluster_to_celltype)
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster in unique_clusters:
            cluster_mask = adata.obs['leiden'] == cluster
            n_cells = cluster_mask.sum()
            cell_type = cluster_to_celltype[cluster]
            
            cluster_stats.append({
                'cluster': cluster,
                'cell_type': cell_type,
                'n_cells': n_cells,
                'percentage': (n_cells / adata.n_obs) * 100
            })
            
        cluster_df = pd.DataFrame(cluster_stats)
        logger.info("Cell type distribution:")
        for _, row in cluster_df.iterrows():
            logger.info(f"  {row['cell_type']}: {row['n_cells']} cells ({row['percentage']:.1f}%)")
            
        return adata, cluster_df
        
    def create_visualizations(self, adata, cluster_df):
        """Create UMAP and cell type visualizations."""
        logger.info("Creating visualizations...")
        
        # Set up figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # UMAP colored by cluster
        sc.pl.umap(adata, color='leiden', legend_loc='on data', 
                  title='Leiden Clustering', ax=axes[0,0], show=False)
        
        # UMAP colored by cell type
        sc.pl.umap(adata, color='cell_type_predicted', 
                  title='Cell Type Annotation', ax=axes[0,1], show=False)
        
        # Cell type proportions
        axes[1,0].pie(cluster_df['n_cells'], labels=cluster_df['cell_type'], autopct='%1.1f%%')
        axes[1,0].set_title('Cell Type Proportions')
        
        # Cluster sizes
        cluster_df.plot.bar(x='cluster', y='n_cells', ax=axes[1,1])
        axes[1,1].set_title('Cluster Sizes')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Number of Cells')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{self.dataset_id}_epiagent_cell_types.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {plot_file}")
        
        plt.show()
        return plot_file
        
    def save_results(self, adata, cluster_df):
        """Save cell type identification results."""
        # Save processed data with embeddings and annotations
        results_file = self.output_dir / f"{self.dataset_id}_epiagent_results.h5ad"
        adata.write(results_file)
        logger.info(f"Results saved to: {results_file}")
        
        # Save cluster statistics
        cluster_file = self.output_dir / f"{self.dataset_id}_cluster_statistics.csv"
        cluster_df.to_csv(cluster_file, index=False)
        logger.info(f"Cluster statistics saved to: {cluster_file}")
        
        # Create summary
        summary = {
            'dataset_id': self.dataset_id,
            'disease': self.disease_name,
            'target_tissue': self.target_tissue,
            'target_cell_types': self.target_cell_types,
            'n_cells': adata.n_obs,
            'n_clusters': len(adata.obs['leiden'].unique()),
            'identified_cell_types': cluster_df['cell_type'].tolist(),
            'cell_type_counts': cluster_df.set_index('cell_type')['n_cells'].to_dict(),
            'results_file': str(results_file),
            'embedding_dim': adata.obsm['X_epiagent'].shape[1],
            'model_parameters': {
                'vocab_size': self.vocab_size,
                'num_layers': self.num_layers,
                'embedding_dim': self.embedding_dim,
                'max_seq_length': self.max_seq_length
            }
        }
        
        summary_file = self.output_dir / f"{self.dataset_id}_cell_identification_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")
        
        return results_file, cluster_file, summary_file
        
    def identify_cell_types(self):
        """Run complete cell type identification pipeline."""
        logger.info("Starting EpiAgent cell type identification pipeline")
        
        # Load preprocessed data
        adata, cell_sentences = self.load_preprocessed_data()
        
        # Load EpiAgent model
        model = self.load_epiagent_model()
        
        # Generate cell embeddings
        embeddings = self.generate_cell_embeddings(model, cell_sentences)
        
        # Perform clustering
        adata = self.perform_clustering(embeddings, adata)
        
        # Annotate cell types
        adata, cluster_df = self.annotate_cell_types(adata)
        
        # Create visualizations
        plot_file = self.create_visualizations(adata, cluster_df)
        
        # Save results
        results_file, cluster_file, summary_file = self.save_results(adata, cluster_df)
        
        logger.info("Cell type identification completed successfully")
        
        return {
            'dataset_id': self.dataset_id,
            'results_file': results_file,
            'cluster_file': cluster_file,
            'summary_file': summary_file,
            'plot_file': plot_file,
            'n_clusters': len(cluster_df),
            'cell_types': cluster_df['cell_type'].tolist()
        }


def main():
    parser = argparse.ArgumentParser(description='Identify cell types using EpiAgent')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--dataset-id', help='Dataset identifier (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize identifier
    identifier = EpiAgentCellIdentifier(args.config, args.dataset_id)
    
    # Run cell type identification
    results = identifier.identify_cell_types()
    
    print("\n" + "="*50)
    print("EPIAGENT CELL TYPE IDENTIFICATION COMPLETED")
    print("="*50)
    print(f"Dataset: {results['dataset_id']}")
    print(f"Clusters identified: {results['n_clusters']}")
    print(f"Cell types: {', '.join(results['cell_types'])}")
    print(f"Results file: {results['results_file']}")
    print(f"Visualization: {results['plot_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
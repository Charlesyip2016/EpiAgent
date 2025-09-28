# GWAS-Enformer-EpiAgent Workflow

A comprehensive computational workflow for functionally validating disease-associated SNPs by linking them to cis-regulatory elements (cCREs) using GWAS data, Enformer predictions, and EpiAgent single-cell analysis.

## Overview

This workflow implements a three-phase integrative analysis approach:

1. **Phase 1: Hypothesis Generation (GWAS & Enformer)**
   - Identify candidate SNPs from GWAS data
   - Predict functional impact using Enformer model
   - Identify candidate enhancer regions

2. **Phase 2: Validation & Contextualization (EpiAgent)**
   - Process single-cell ATAC-seq data using EpiAgent methodology
   - Identify cell types using EpiAgent foundation model
   - Validate enhancer activity in disease-relevant cell types

3. **Phase 3: Integration & Reporting**
   - Generate comprehensive analysis report
   - Provide mechanistic hypotheses
   - Suggest experimental validation strategies

## Installation

### 1. Environment Setup

Create and activate the conda environment:

```bash
cd gwas_enformer_epiagent_workflow/environment
chmod +x setup_environment.sh
./setup_environment.sh
conda activate gwas_enformer_epiagent
```

### 2. Manual Dependencies (if needed)

If the automatic setup fails, install dependencies manually:

```bash
conda create -n gwas_enformer_epiagent python=3.11
conda activate gwas_enformer_epiagent

# Core dependencies
conda install pandas numpy scipy matplotlib seaborn scikit-learn
conda install jupyter notebook jupyterlab
conda install -c bioconda pyfaidx pysam bedtools plink
conda install -c conda-forge scanpy anndata
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Python packages
pip install tensorflow tensorflow-hub
pip install episcanpy
pip install epiagent
pip install transformers
pip install POT faiss-cpu flash-attn
pip install biopython plotly kaleido
```

## Quick Start

### 1. Configure Analysis Parameters

Edit the configuration file:

```bash
vim config/config.ini
```

Update the following sections for your analysis:
- `[disease]`: Disease name, GWAS study ID, target tissue
- `[data]`: Data file paths, dataset IDs
- `[analysis]`: Analysis thresholds and parameters

### 2. Run Complete Workflow

```bash
python scripts/run_workflow.py --config config/config.ini
```

### 3. Run Individual Phases

```bash
# Phase 1 only
python scripts/run_workflow.py --config config/config.ini --phases 1

# Phase 2 only  
python scripts/run_workflow.py --config config/config.ini --phases 2

# Phase 3 only
python scripts/run_workflow.py --config config/config.ini --phases 3
```

## Detailed Usage

### Phase 1: GWAS & Enformer Analysis

#### Step 1.1: GWAS Data Analysis
```bash
cd phase1_gwas_enformer
python gwas_analysis.py --config ../config/config.ini --disease "Alzheimer's disease"
```

**Outputs:**
- `{disease}_gwas_summary.tsv`: GWAS summary statistics
- `{disease}_candidate_snps.csv`: SNPs in linkage disequilibrium
- `{disease}_analysis_summary.json`: Analysis metadata

#### Step 1.2: Sequence Preparation
```bash
python prepare_enformer_sequences.py --config ../config/config.ini --target-snp rs1000000
```

**Outputs:**
- `{snp}_reference_sequence.fasta`: Reference allele sequence
- `{snp}_variant_sequence.fasta`: Risk allele sequence
- `{snp}_sequence_metadata.json`: Sequence metadata

#### Step 1.3: Enformer Predictions
```bash
python run_enformer_predictions.py --config ../config/config.ini --target-snp rs1000000
```

**Outputs:**
- `{snp}_enformer_predictions.h5`: Raw prediction arrays
- `{snp}_enformer_summary.json`: Summary statistics

#### Step 1.4: Enhancer Identification
```bash
python identify_functional_enhancer.py --config ../config/config.ini --target-snp rs1000000
```

**Outputs:**
- `{snp}_candidate_enhancers.csv`: Prioritized candidate enhancers
- `{snp}_enhancer_analysis_summary.json`: Top enhancer coordinates
- `{snp}_enformer_track_differences.png`: Visualization

### Phase 2: EpiAgent Validation

#### Step 2.1: Data Preprocessing
```bash
cd phase2_epiagent_validation
python epiagent_preprocessing.py --config ../config/config.ini --dataset-id GSE12345
```

**Outputs:**
- `{dataset}_processed_epiagent.h5ad`: Processed AnnData object
- `{dataset}_cell_sentences.json`: EpiAgent-compatible cell sentences
- `{dataset}_preprocessing_summary.json`: Processing metadata

#### Step 2.2: Cell Type Identification
```bash
python epiagent_cell_identification.py --config ../config/config.ini --dataset-id GSE12345
```

**Outputs:**
- `{dataset}_epiagent_results.h5ad`: Results with embeddings and cell types
- `{dataset}_cluster_statistics.csv`: Cluster information
- `{dataset}_epiagent_cell_types.png`: UMAP visualization

#### Step 2.3: Enhancer Validation
```bash
python validate_enhancer_specificity.py --config ../config/config.ini --target-snp rs1000000
```

**Outputs:**
- `{snp}_enhancer_validation_results.json`: Detailed validation results
- `{snp}_celltype_accessibility_summary.csv`: Accessibility by cell type
- `{snp}_enhancer_validation.png`: Validation plots

### Phase 3: Integration & Reporting

#### Step 3.1: Generate Final Report
```bash
cd phase3_integration_reporting  
python generate_final_report.py --config ../config/config.ini --target-snp rs1000000
```

**Outputs:**
- `{snp}_final_report.md`: Comprehensive markdown report
- `{snp}_final_report.html`: HTML version (if markdown package available)

## Configuration Reference

### Disease Section
```ini
[disease]
name = "Alzheimer's disease"
gwas_catalog_id = "GCST90027158"
target_tissue = "prefrontal cortex"  
target_cell_types = ["microglia", "astrocytes", "neurons"]
```

### GWAS Analysis
```ini
[gwas]
ld_threshold = 0.8
reference_panel = "1000GP_Phase3"
p_value_threshold = 5e-8
```

### Enformer Parameters
```ini
[enformer]
model_url = "https://tfhub.dev/deepmind/enformer/1"
sequence_length = 196608
prediction_tracks = ["CAGE", "DNase", "ATAC-seq", "H3K27ac"]
cell_lines_of_interest = ["K562", "GM12878", "HepG2"]
```

### EpiAgent Parameters
```ini
[epiagent]
model_path = "./model/pretrained_EpiAgent.pth"
ccre_reference_file = "./data/ccre_coordinates_hg38.bed"
max_sequence_length = 8192
vocab_size = 1355449
num_layers = 18
embedding_dim = 512
num_attention_heads = 8
```

### Analysis Thresholds
```ini
[analysis]
n_top_snps = 10
effect_size_threshold = 0.1
cell_type_specificity_threshold = 0.5
```

## Expected Runtime

- **Phase 1**: 15-30 minutes
  - GWAS analysis: 2-5 minutes
  - Sequence preparation: 1-2 minutes  
  - Enformer predictions: 10-20 minutes
  - Enhancer identification: 2-3 minutes

- **Phase 2**: 20-45 minutes
  - Data preprocessing: 5-10 minutes
  - EpiAgent inference: 10-30 minutes (depends on dataset size)
  - Enhancer validation: 5-10 minutes

- **Phase 3**: 2-5 minutes
  - Report generation: 2-5 minutes

**Total**: ~40-80 minutes for complete workflow

## Output Structure

```
data/results/
├── {disease}_gwas_summary.tsv
├── {disease}_candidate_snps.csv
├── {disease}_analysis_summary.json
├── {snp}_reference_sequence.fasta
├── {snp}_variant_sequence.fasta
├── {snp}_sequence_metadata.json
├── {snp}_enformer_predictions.h5
├── {snp}_enformer_summary.json
├── {snp}_candidate_enhancers.csv
├── {snp}_enhancer_analysis_summary.json
├── {snp}_enformer_track_differences.png
├── {dataset}_processed_epiagent.h5ad
├── {dataset}_cell_sentences.json
├── {dataset}_preprocessing_summary.json
├── {dataset}_epiagent_results.h5ad
├── {dataset}_cluster_statistics.csv
├── {dataset}_epiagent_cell_types.png
├── {snp}_enhancer_validation_results.json
├── {snp}_celltype_accessibility_summary.csv
├── {snp}_enhancer_validation.png
├── {snp}_accessibility_distribution.png
├── {snp}_final_report.md
├── {snp}_final_report.html
└── workflow_state.json
```

## Troubleshooting

### Common Issues

1. **Environment Setup Issues**
   ```bash
   # If conda environment creation fails
   conda clean --all
   conda update conda
   # Then retry setup
   ```

2. **EpiAgent Model Not Found**
   ```bash
   # Download pretrained model manually
   mkdir -p model
   # Download from: https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX
   ```

3. **Memory Issues with Large Datasets**
   ```bash
   # Reduce dataset size in preprocessing
   # Or run on high-memory machine
   ```

4. **CUDA/GPU Issues**
   ```bash
   # Install CPU-only versions
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install faiss-cpu
   ```

### Log Files

Check workflow logs in the console output or redirect to file:
```bash
python scripts/run_workflow.py --config config/config.ini > workflow.log 2>&1
```

## Citation

If you use this workflow in your research, please cite:

1. **EpiAgent**: Chen X, Li K, Cui X, Wang Z, Jiang Q, Lin J, Li Z, Gao Z, Jiang R. EpiAgent: Foundation model for single-cell epigenomic data. bioRxiv. 2024:2024-12.

2. **Enformer**: Avsec, Ž., Agarwal, V., Visentin, D. et al. Effective gene expression prediction from sequence by integrating long-range interactions. Nat Methods 18, 1196–1203 (2021).

## Contact

For questions about this workflow, please open an issue on the GitHub repository or contact the EpiAgent authors.

## License

This workflow is released under the same license as the EpiAgent repository (MIT License).
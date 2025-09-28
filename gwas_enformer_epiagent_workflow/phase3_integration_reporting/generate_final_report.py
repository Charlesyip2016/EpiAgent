#!/usr/bin/env python3
"""
Phase 3.1: Generate Final Report

This script generates a comprehensive markdown report summarizing all findings
from the GWAS-Enformer-EpiAgent integrative analysis workflow.

Usage:
    python generate_final_report.py --config ../config/config.ini --target-snp rs1000000
"""

import argparse
import configparser
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, config_file, target_snp=None):
        """Initialize report generator with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.disease_name = self.config['disease']['name']
        self.target_tissue = self.config['disease']['target_tissue']
        self.target_cell_types = self.config['disease']['target_cell_types'].split(', ')
        
        self.output_dir = Path(self.config['data']['output_dir'])
        self.processed_dir = Path(self.config['data']['processed_data_dir'])
        
        self.target_snp = target_snp
        self.dataset_id = self.config['data']['scatac_dataset_id']
        
    def load_analysis_results(self):
        """Load results from all analysis phases."""
        results = {}
        
        # Phase 1 results
        try:
            # GWAS analysis summary
            gwas_summary_file = self.output_dir / f"{self.disease_name.replace(' ', '_')}_analysis_summary.json"
            if gwas_summary_file.exists():
                with open(gwas_summary_file, 'r') as f:
                    results['gwas_analysis'] = json.load(f)
                    
            # Enhancer analysis
            if self.target_snp is None and 'gwas_analysis' in results:
                self.target_snp = results['gwas_analysis']['lead_snp']
                
            if self.target_snp:
                enhancer_summary_file = self.output_dir / f"{self.target_snp}_enhancer_analysis_summary.json"
                if enhancer_summary_file.exists():
                    with open(enhancer_summary_file, 'r') as f:
                        results['enhancer_analysis'] = json.load(f)
                        
                # Enformer results
                enformer_summary_file = self.output_dir / f"{self.target_snp}_enformer_summary.json"
                if enformer_summary_file.exists():
                    with open(enformer_summary_file, 'r') as f:
                        results['enformer_results'] = json.load(f)
                        
        except Exception as e:
            logger.warning(f"Could not load Phase 1 results: {e}")
            
        # Phase 2 results
        try:
            # EpiAgent cell identification
            cell_id_summary_file = self.output_dir / f"{self.dataset_id}_cell_identification_summary.json"
            if cell_id_summary_file.exists():
                with open(cell_id_summary_file, 'r') as f:
                    results['cell_identification'] = json.load(f)
                    
            # Enhancer validation
            if self.target_snp:
                validation_file = self.output_dir / f"{self.target_snp}_enhancer_validation_results.json"
                if validation_file.exists():
                    with open(validation_file, 'r') as f:
                        results['enhancer_validation'] = json.load(f)
                        
        except Exception as e:
            logger.warning(f"Could not load Phase 2 results: {e}")
            
        if not results:
            raise FileNotFoundError("No analysis results found. Please run the analysis pipeline first.")
            
        logger.info(f"Loaded results for phases: {', '.join(results.keys())}")
        return results
        
    def find_analysis_plots(self):
        """Find all generated plots for inclusion in report."""
        plots = {}
        
        if self.target_snp:
            # Phase 1 plots
            enformer_plot = self.output_dir / f"{self.target_snp}_enformer_track_differences.png"
            if enformer_plot.exists():
                plots['enformer_differences'] = enformer_plot
                
        # Phase 2 plots
        cell_types_plot = self.output_dir / f"{self.dataset_id}_epiagent_cell_types.png"
        if cell_types_plot.exists():
            plots['cell_types'] = cell_types_plot
            
        if self.target_snp:
            validation_plot = self.output_dir / f"{self.target_snp}_enhancer_validation.png"
            if validation_plot.exists():
                plots['enhancer_validation'] = validation_plot
                
            distribution_plot = self.output_dir / f"{self.target_snp}_accessibility_distribution.png"
            if distribution_plot.exists():
                plots['accessibility_distribution'] = distribution_plot
                
        logger.info(f"Found {len(plots)} plots for report")
        return plots
        
    def encode_image_base64(self, image_path):
        """Encode image as base64 for embedding in markdown."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.warning(f"Could not encode image {image_path}: {e}")
            return None
            
    def generate_markdown_report(self, results, plots):
        """Generate comprehensive markdown report."""
        report = []
        
        # Header
        report.append("# GWAS-Enformer-EpiAgent Integrative Analysis Report")
        report.append("")
        report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        if 'gwas_analysis' in results:
            gwas = results['gwas_analysis']
            report.append(f"**Disease Studied:** {gwas['disease']}")
            report.append(f"**Lead SNP:** {gwas['lead_snp']} (p-value: {gwas['lead_snp_pvalue']:.2e})")
            report.append(f"**Chromosome:** {gwas['lead_snp_chr']}, Position: {gwas['lead_snp_pos']:,}")
            report.append(f"**Candidate SNPs Analyzed:** {gwas['n_candidate_snps']}")
        else:
            report.append(f"**Disease Studied:** {self.disease_name}")
            if self.target_snp:
                report.append(f"**Target SNP:** {self.target_snp}")
            
        if 'cell_identification' in results:
            cell_id = results['cell_identification']
            report.append(f"**Target Tissue:** {cell_id['target_tissue']}")
            report.append(f"**Cells Analyzed:** {cell_id['n_cells']:,}")
            report.append(f"**Cell Types Identified:** {len(cell_id['identified_cell_types'])}")
            
        report.append("")
        
        # Introduction
        report.append("## Introduction")
        report.append("")
        report.append(f"This report presents the results of an integrative computational analysis to ")
        report.append(f"functionally validate disease-associated genetic variants for **{self.disease_name}**. ")
        report.append(f"The analysis employed a three-phase approach:")
        report.append("")
        report.append("1. **Hypothesis Generation**: GWAS data analysis and Enformer predictions")
        report.append("2. **Validation & Contextualization**: EpiAgent-based single-cell analysis")
        report.append("3. **Integration**: Comprehensive synthesis of findings")
        report.append("")
        
        # Phase 1: GWAS & Enformer Results
        report.append("## Phase 1: Hypothesis Generation (GWAS & Enformer)")
        report.append("")
        
        if 'gwas_analysis' in results:
            report.append("### GWAS Analysis Results")
            gwas = results['gwas_analysis']
            
            report.append("")
            report.append(f"- **Lead SNP identified:** {gwas['lead_snp']}")
            report.append(f"- **Genomic location:** chr{gwas['lead_snp_chr']}:{gwas['lead_snp_pos']:,}")
            report.append(f"- **Statistical significance:** p = {gwas['lead_snp_pvalue']:.2e}")
            report.append(f"- **Linkage disequilibrium threshold:** r² > {gwas['ld_threshold']}")
            report.append(f"- **Total candidate SNPs:** {gwas['n_candidate_snps']}")
            report.append("")
        
        if 'enformer_results' in results:
            report.append("### Enformer Prediction Results")
            enformer = results['enformer_results']
            
            report.append("")
            report.append(f"The Enformer model was used to predict the functional impact of {enformer['target_snp']} ")
            report.append(f"on chromatin accessibility and histone modifications across a 196,608 bp sequence window.")
            report.append("")
            report.append("**Tracks analyzed:**")
            for track in enformer['prediction_tracks']:
                if f'{track}_max_log2fc' in enformer:
                    max_effect = enformer[f'{track}_max_log2fc']
                    mean_effect = enformer[f'{track}_mean_log2fc']
                    report.append(f"- **{track}**: Maximum effect size = {max_effect:.3f}, Mean = {mean_effect:.3f}")
            report.append("")
            
            # Add Enformer plot if available
            if 'enformer_differences' in plots:
                report.append("**Figure 1: Enformer Prediction Differences**")
                report.append("")
                report.append(f"![Enformer Differences]({plots['enformer_differences']})")
                report.append("")
                report.append("*Figure 1: Log2 fold change predictions across the genomic region for different ")
                report.append("chromatin accessibility and histone modification tracks. Red vertical line indicates ")
                report.append("SNP position, highlighted regions show candidate enhancers.*")
                report.append("")
                
        if 'enhancer_analysis' in results:
            report.append("### Candidate Enhancer Identification")
            enhancer = results['enhancer_analysis']
            
            if enhancer['candidate_enhancer']:
                enh = enhancer['candidate_enhancer']
                report.append("")
                report.append("**Identified Candidate Enhancer:**")
                report.append("")
                report.append(f"- **Genomic coordinates:** chr{enh['chromosome']}:{enh['start']:,}-{enh['end']:,}")
                report.append(f"- **Effect size:** {enh['max_effect_size']:.3f}")
                report.append(f"- **Effect direction:** {enh['effect_direction']}")
                report.append(f"- **Track type:** {enh['track_type']}")
                report.append(f"- **Distance from SNP:** ~{abs(enh['center'] - enhancer['snp_position']):,} bp")
                report.append("")
            else:
                report.append("")
                report.append("**No significant candidate enhancers were identified** meeting the analysis thresholds.")
                report.append("")
                
        # Phase 2: EpiAgent Results
        report.append("## Phase 2: Validation & Contextualization (EpiAgent)")
        report.append("")
        
        if 'cell_identification' in results:
            report.append("### Single-Cell Analysis Results")
            cell_id = results['cell_identification']
            
            report.append("")
            report.append(f"Single-cell ATAC-seq data from **{cell_id['target_tissue']}** was analyzed using ")
            report.append(f"the EpiAgent foundation model to identify cell types and validate enhancer activity.")
            report.append("")
            report.append("**Dataset characteristics:**")
            report.append(f"- **Number of cells:** {cell_id['n_cells']:,}")
            report.append(f"- **Number of cCREs:** {cell_id.get('n_ccres', 'N/A'):,}")
            report.append(f"- **Clusters identified:** {cell_id['n_clusters']}")
            report.append("")
            
            report.append("**Cell type distribution:**")
            if 'cell_type_counts' in cell_id:
                for cell_type, count in cell_id['cell_type_counts'].items():
                    percentage = (count / cell_id['n_cells']) * 100
                    is_target = " ⭐" if cell_type in self.target_cell_types else ""
                    report.append(f"- **{cell_type}:** {count:,} cells ({percentage:.1f}%){is_target}")
            else:
                for cell_type in cell_id['identified_cell_types']:
                    is_target = " ⭐" if cell_type in self.target_cell_types else ""
                    report.append(f"- {cell_type}{is_target}")
                    
            report.append("")
            report.append("*⭐ indicates cell types relevant to the studied disease*")
            report.append("")
            
            # Add cell types plot if available
            if 'cell_types' in plots:
                report.append("**Figure 2: Cell Type Identification**")
                report.append("")
                report.append(f"![Cell Types]({plots['cell_types']})")
                report.append("")
                report.append("*Figure 2: EpiAgent-based cell type identification. Top panels show UMAP ")
                report.append("visualization colored by clusters and predicted cell types. Bottom panels ")
                report.append("show cell type proportions and cluster sizes.*")
                report.append("")
                
        if 'enhancer_validation' in results:
            report.append("### Enhancer Validation Results")
            validation = results['enhancer_validation']
            
            report.append("")
            if validation['selected_ccre']:
                ccre = validation['selected_ccre']
                report.append(f"**Validated cCRE:** {ccre['ccre_id']}")
                report.append(f"**Coordinates:** {ccre['coordinates']['chromosome']}:")
                report.append(f"{ccre['coordinates']['start']:,}-{ccre['coordinates']['end']:,}")
                
                if ccre['overlap_with_enhancer'] > 0:
                    report.append(f"**Overlap with candidate enhancer:** {ccre['overlap_with_enhancer']:,} bp")
                else:
                    report.append("**Note:** Selected closest cCRE (no direct overlap found)")
                    
                report.append("")
                
            # Cell-type specific accessibility
            if 'celltype_accessibility' in validation:
                report.append("**Cell-type specific accessibility:**")
                report.append("")
                
                accessibility = validation['celltype_accessibility']
                sorted_celltypes = sorted(accessibility.items(), 
                                        key=lambda x: x[1]['mean'], reverse=True)
                
                for cell_type, stats in sorted_celltypes:
                    is_target = " ⭐" if cell_type in self.target_cell_types else ""
                    report.append(f"- **{cell_type}{is_target}:** Mean accessibility = {stats['mean']:.3f}, ")
                    report.append(f"  Accessibility rate = {stats['accessibility_rate']:.1%} ")
                    report.append(f"  ({stats['n_cells']:,} cells)")
                    
                report.append("")
                
            # Specificity analysis
            if 'specificity_analysis' in validation:
                specificity = validation['specificity_analysis']
                
                if specificity.get('most_specific_celltype'):
                    most_specific = specificity['most_specific_celltype']
                    is_significant = specificity.get('significant', False)
                    
                    report.append(f"**Most specific cell type:** {most_specific}")
                    if is_significant:
                        report.append(f"**Statistical significance:** p = {specificity['pvalue']:.3e} ✓")
                        report.append(f"**Effect size:** {specificity['effect_size']:.3f}")
                    else:
                        report.append("**Statistical significance:** Not significant")
                        
                    report.append("")
                    
            # Disease relevance
            if 'disease_relevance' in validation:
                relevance = validation['disease_relevance']
                
                report.append("**Disease Relevance Assessment:**")
                report.append("")
                
                if relevance['is_disease_relevant']:
                    report.append("✅ **The candidate enhancer shows activity in disease-relevant cell types.**")
                else:
                    report.append("⚠️ **The candidate enhancer does not show preferential activity in known disease-relevant cell types.**")
                    
                report.append("")
                report.append(f"*{relevance['explanation']}*")
                report.append("")
                
            # Add validation plots if available
            if 'enhancer_validation' in plots:
                report.append("**Figure 3: Enhancer Validation Analysis**")
                report.append("")
                report.append(f"![Enhancer Validation]({plots['enhancer_validation']})")
                report.append("")
                report.append("*Figure 3: Cell-type specific accessibility analysis of the candidate enhancer. ")
                report.append("Bar plots show mean accessibility and accessibility rates across cell types. ")
                report.append("UMAP plots show spatial distribution of accessibility and cell types.*")
                report.append("")
                
            if 'accessibility_distribution' in plots:
                report.append("**Figure 4: Accessibility Distribution**")
                report.append("")
                report.append(f"![Accessibility Distribution]({plots['accessibility_distribution']})")
                report.append("")
                report.append("*Figure 4: Distribution of accessibility scores across cell types ")
                report.append("for the validated cCRE, showing cell-type specific patterns.*")
                report.append("")
                
        # Integrated Conclusion
        report.append("## Integrated Conclusion")
        report.append("")
        
        # Generate conclusion based on available results
        if 'gwas_analysis' in results and 'enhancer_validation' in results:
            gwas = results['gwas_analysis']
            validation = results['enhancer_validation']
            
            conclusion = f"Based on this integrative analysis, **{gwas['lead_snp']}** "
            
            if validation['disease_relevance']['is_disease_relevant']:
                if 'enhancer_analysis' in results and results['enhancer_analysis']['candidate_enhancer']:
                    enh = results['enhancer_analysis']['candidate_enhancer']
                    most_specific = validation['specificity_analysis'].get('most_specific_celltype', 'specific cell types')
                    
                    conclusion += f"likely contributes to **{gwas['disease']}** risk by altering the activity of "
                    conclusion += f"a regulatory enhancer located at **chr{enh['chromosome']}:{enh['start']:,}-{enh['end']:,}** "
                    conclusion += f"within **{most_specific}** cells in the {validation['target_tissue']}."
                    
                    report.append(conclusion)
                    report.append("")
                    
                    report.append("**Key supporting evidence:**")
                    report.append(f"1. The SNP shows genome-wide significant association with {gwas['disease']} (p = {gwas['lead_snp_pvalue']:.2e})")
                    report.append(f"2. Enformer predicts functional impact on chromatin accessibility (effect size: {enh['max_effect_size']:.3f})")
                    report.append(f"3. The enhancer region shows {enh['effect_direction']}d activity in the variant compared to reference")
                    report.append(f"4. Single-cell analysis confirms cell-type specific activity in {most_specific}")
                    if most_specific in self.target_cell_types:
                        report.append(f"5. {most_specific} are known to be involved in {gwas['disease']} pathogenesis")
                        
                else:
                    conclusion += f"may contribute to **{gwas['disease']}** risk, though no specific enhancer "
                    conclusion += f"could be confidently identified in this analysis."
                    report.append(conclusion)
                    
            else:
                conclusion += f"shows potential functional effects, but the analysis did not provide strong "
                conclusion += f"evidence for a specific mechanism in **{gwas['disease']}**-relevant cell types."
                report.append(conclusion)
                
        else:
            report.append("The integrative analysis workflow was partially completed. ")
            report.append("Complete results from all phases would provide stronger evidence ")
            report.append("for the functional role of the analyzed genetic variants.")
            
        report.append("")
        
        # Future Directions
        report.append("## Future Directions")
        report.append("")
        report.append("To further validate these computational predictions, we recommend:")
        report.append("")
        report.append("### Experimental Validation")
        report.append("")
        
        if 'enhancer_analysis' in results and results['enhancer_analysis']['candidate_enhancer']:
            enh = results['enhancer_analysis']['candidate_enhancer']
            coords = f"chr{enh['chromosome']}:{enh['start']}-{enh['end']}"
            
            report.append(f"1. **CRISPRi/CRISPRa experiments:** Design guide RNAs targeting the enhancer region ")
            report.append(f"   ({coords}) to directly test its regulatory function")
            report.append("")
            report.append("2. **Reporter assays:** Clone the enhancer region into luciferase or GFP reporter ")
            report.append("   constructs to measure both reference and variant activity")
            report.append("")
            report.append("3. **Chromatin conformation capture (3C/4C-seq):** Map physical interactions ")
            report.append("   between the enhancer and potential target genes")
            report.append("")
            
            if 'enhancer_validation' in results and results['enhancer_validation']['specificity_analysis'].get('most_specific_celltype'):
                cell_type = results['enhancer_validation']['specificity_analysis']['most_specific_celltype']
                report.append(f"4. **Cell-type specific experiments:** Focus validation experiments on ")
                report.append(f"   {cell_type} or develop {cell_type}-specific model systems")
                report.append("")
                
        report.append("### Computational Extensions")
        report.append("")
        report.append("1. **Gene target identification:** Use additional methods (ABC model, ")
        report.append("   activity-by-contact) to identify likely target genes")
        report.append("")
        report.append("2. **Multi-tissue analysis:** Extend the analysis to additional tissues ")
        report.append("   and cell types relevant to the disease")
        report.append("")
        report.append("3. **Population genetics:** Analyze allele frequencies and evolutionary ")
        report.append("   signatures around the variant")
        report.append("")
        report.append("4. **Pathway analysis:** Integrate with gene expression data to understand ")
        report.append("   downstream molecular pathways")
        report.append("")
        
        # Methods Summary
        report.append("## Methods Summary")
        report.append("")
        report.append("This analysis employed a three-phase computational workflow:")
        report.append("")
        report.append("**Phase 1 (Hypothesis Generation):**")
        report.append("- GWAS summary statistics analysis for lead SNP identification")
        report.append("- Linkage disequilibrium mapping using 1000 Genomes reference")
        report.append("- Enformer deep learning model for chromatin accessibility prediction")
        report.append("- Candidate enhancer identification based on predicted functional impact")
        report.append("")
        report.append("**Phase 2 (Validation & Contextualization):**")
        report.append("- Single-cell ATAC-seq data preprocessing following EpiAgent methodology")
        report.append("- TF-IDF transformation and cell sentence generation")
        report.append("- EpiAgent foundation model for cell embedding and type identification")
        report.append("- Cell-type specific enhancer accessibility analysis")
        report.append("")
        report.append("**Phase 3 (Integration):**")
        report.append("- Comprehensive synthesis of computational predictions")
        report.append("- Statistical testing for cell-type specificity")
        report.append("- Disease relevance assessment")
        report.append("- Evidence integration and mechanistic hypothesis generation")
        report.append("")
        
        # Technical Details
        if results:
            report.append("## Technical Details")
            report.append("")
            
            if 'enformer_results' in results:
                report.append("**Enformer Analysis:**")
                report.append("- Sequence length: 196,608 bp")
                report.append("- Model: DeepMind Enformer from TensorFlow Hub")
                report.append(f"- Tracks analyzed: {', '.join(results['enformer_results']['prediction_tracks'])}")
                report.append("")
                
            if 'cell_identification' in results:
                cell_id = results['cell_identification']
                params = cell_id.get('model_parameters', {})
                
                report.append("**EpiAgent Analysis:**")
                report.append(f"- Model vocabulary size: {params.get('vocab_size', 'N/A'):,}")
                report.append(f"- Embedding dimensions: {params.get('embedding_dim', 'N/A')}")
                report.append(f"- Transformer layers: {params.get('num_layers', 'N/A')}")
                report.append(f"- Maximum sequence length: {params.get('max_seq_length', 'N/A')}")
                report.append("")
                
            report.append("**Statistical Methods:**")
            report.append("- Mann-Whitney U tests for cell-type accessibility comparisons")
            report.append("- Leiden clustering for unsupervised cell type identification")
            report.append("- UMAP for dimensionality reduction and visualization")
            report.append("")
            
        # Footer
        report.append("---")
        report.append("")
        report.append(f"*Report generated by GWAS-Enformer-EpiAgent Workflow v1.0*")
        report.append(f"*Analysis completed on {datetime.now().strftime('%Y-%m-%d')}*")
        
        return "\\n".join(report)
        
    def save_report(self, report_content):
        """Save the markdown report to file."""
        if self.target_snp:
            report_file = self.output_dir / f"{self.target_snp}_final_report.md"
        else:
            report_file = self.output_dir / f"{self.disease_name.replace(' ', '_')}_final_report.md"
            
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Final report saved to: {report_file}")
        
        # Also save as HTML for better viewing
        try:
            import markdown
            html_content = markdown.markdown(report_content)
            html_file = report_file.with_suffix('.html')
            
            with open(html_file, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>GWAS-Enformer-EpiAgent Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
                """)
                
            logger.info(f"HTML report saved to: {html_file}")
            return report_file, html_file
            
        except ImportError:
            logger.warning("markdown package not available, HTML report not generated")
            return report_file, None
            
    def generate_report(self):
        """Generate complete final report."""
        logger.info("Starting report generation")
        
        # Load all analysis results
        results = self.load_analysis_results()
        
        # Find analysis plots
        plots = self.find_analysis_plots()
        
        # Generate markdown report
        report_content = self.generate_markdown_report(results, plots)
        
        # Save report
        report_files = self.save_report(report_content)
        
        logger.info("Report generation completed successfully")
        
        return {
            'markdown_file': report_files[0],
            'html_file': report_files[1] if len(report_files) > 1 else None,
            'n_phases_included': len(results),
            'plots_included': len(plots),
            'target_snp': self.target_snp,
            'disease': self.disease_name
        }


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis report')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--target-snp', help='Target SNP rsID')
    
    args = parser.parse_args()
    
    # Initialize report generator
    generator = ReportGenerator(args.config, args.target_snp)
    
    # Generate report
    results = generator.generate_report()
    
    print("\\n" + "="*50)
    print("FINAL REPORT GENERATION COMPLETED")
    print("="*50)
    print(f"Disease: {results['disease']}")
    if results['target_snp']:
        print(f"Target SNP: {results['target_snp']}")
    print(f"Analysis phases included: {results['n_phases_included']}")
    print(f"Plots included: {results['plots_included']}")
    print(f"Markdown report: {results['markdown_file']}")
    if results['html_file']:
        print(f"HTML report: {results['html_file']}")
    print("="*50)


if __name__ == "__main__":
    main()
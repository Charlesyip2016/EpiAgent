#!/usr/bin/env python3
"""
Main workflow orchestrator for GWAS-Enformer-EpiAgent analysis

This script runs the complete analysis pipeline from GWAS data through
Enformer predictions to EpiAgent validation and final reporting.

Usage:
    python run_workflow.py --config config/config.ini --disease "Alzheimer's disease"
"""

import argparse
import configparser
import sys
import subprocess
from pathlib import Path
import logging
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    def __init__(self, config_file, disease_name=None, target_snp=None, dataset_id=None):
        """Initialize workflow orchestrator."""
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # Override config parameters if provided
        if disease_name:
            self.config['disease']['name'] = disease_name
        if dataset_id:
            self.config['data']['scatac_dataset_id'] = dataset_id
            
        self.disease_name = self.config['disease']['name']
        self.target_snp = target_snp
        
        # Set up paths
        self.workflow_dir = self.config_file.parent.parent
        self.phase1_dir = self.workflow_dir / 'phase1_gwas_enformer'
        self.phase2_dir = self.workflow_dir / 'phase2_epiagent_validation'
        self.phase3_dir = self.workflow_dir / 'phase3_integration_reporting'
        self.output_dir = Path(self.config['data']['output_dir'])
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track workflow state
        self.workflow_state = {
            'phase1_gwas': False,
            'phase1_sequences': False,
            'phase1_enformer': False,
            'phase1_enhancer': False,
            'phase2_preprocessing': False,
            'phase2_cell_identification': False,
            'phase2_validation': False,
            'phase3_report': False
        }
        
    def run_script(self, script_path, args=None, required=True):
        """Run a Python script with error handling."""
        script_path = Path(script_path)
        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            if required:
                raise FileNotFoundError(error_msg)
            else:
                logger.warning(error_msg)
                return False, None
                
        cmd = [sys.executable, str(script_path), '--config', str(self.config_file)]
        
        if args:
            cmd.extend(args)
            
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.time()
            
            logger.info(f"Script completed successfully in {end_time - start_time:.1f}s")
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Script failed with return code {e.returncode}"
            logger.error(error_msg)
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            
            if required:
                raise RuntimeError(error_msg)
            else:
                return False, e.stderr
                
    def run_phase1(self):
        """Run Phase 1: GWAS & Enformer analysis."""
        logger.info("=" * 60)
        logger.info("STARTING PHASE 1: GWAS & ENFORMER ANALYSIS")
        logger.info("=" * 60)
        
        # Step 1.1: GWAS Analysis
        logger.info("Step 1.1: Analyzing GWAS data...")
        success, output = self.run_script(
            self.phase1_dir / 'gwas_analysis.py',
            args=['--disease', self.disease_name]
        )
        if success:
            self.workflow_state['phase1_gwas'] = True
            logger.info("✓ GWAS analysis completed")
        else:
            raise RuntimeError("GWAS analysis failed")
            
        # Step 1.2: Prepare sequences
        logger.info("Step 1.2: Preparing sequences for Enformer...")
        args = []
        if self.target_snp:
            args.extend(['--target-snp', self.target_snp])
            
        success, output = self.run_script(
            self.phase1_dir / 'prepare_enformer_sequences.py',
            args=args
        )
        if success:
            self.workflow_state['phase1_sequences'] = True
            logger.info("✓ Sequence preparation completed")
        else:
            raise RuntimeError("Sequence preparation failed")
            
        # Step 1.3: Run Enformer
        logger.info("Step 1.3: Running Enformer predictions...")
        args = []
        if self.target_snp:
            args.extend(['--target-snp', self.target_snp])
            
        success, output = self.run_script(
            self.phase1_dir / 'run_enformer_predictions.py',
            args=args
        )
        if success:
            self.workflow_state['phase1_enformer'] = True
            logger.info("✓ Enformer predictions completed")
        else:
            raise RuntimeError("Enformer predictions failed")
            
        # Step 1.4: Identify enhancers
        logger.info("Step 1.4: Identifying functional enhancers...")
        args = []
        if self.target_snp:
            args.extend(['--target-snp', self.target_snp])
            
        success, output = self.run_script(
            self.phase1_dir / 'identify_functional_enhancer.py',
            args=args
        )
        if success:
            self.workflow_state['phase1_enhancer'] = True
            logger.info("✓ Enhancer identification completed")
        else:
            raise RuntimeError("Enhancer identification failed")
            
        logger.info("✅ PHASE 1 COMPLETED SUCCESSFULLY")
        return True
        
    def run_phase2(self):
        """Run Phase 2: EpiAgent validation."""
        logger.info("=" * 60)
        logger.info("STARTING PHASE 2: EPIAGENT VALIDATION")
        logger.info("=" * 60)
        
        dataset_id = self.config['data']['scatac_dataset_id']
        
        # Step 2.1: Data preprocessing
        logger.info("Step 2.1: Preprocessing scATAC-seq data...")
        success, output = self.run_script(
            self.phase2_dir / 'epiagent_preprocessing.py',
            args=['--dataset-id', dataset_id]
        )
        if success:
            self.workflow_state['phase2_preprocessing'] = True
            logger.info("✓ EpiAgent preprocessing completed")
        else:
            raise RuntimeError("EpiAgent preprocessing failed")
            
        # Step 2.2: Cell identification
        logger.info("Step 2.2: Running EpiAgent cell type identification...")
        success, output = self.run_script(
            self.phase2_dir / 'epiagent_cell_identification.py',
            args=['--dataset-id', dataset_id]
        )
        if success:
            self.workflow_state['phase2_cell_identification'] = True
            logger.info("✓ Cell type identification completed")
        else:
            raise RuntimeError("Cell type identification failed")
            
        # Step 2.3: Enhancer validation
        logger.info("Step 2.3: Validating enhancer specificity...")
        args = ['--dataset-id', dataset_id]
        if self.target_snp:
            args.extend(['--target-snp', self.target_snp])
            
        success, output = self.run_script(
            self.phase2_dir / 'validate_enhancer_specificity.py',
            args=args
        )
        if success:
            self.workflow_state['phase2_validation'] = True
            logger.info("✓ Enhancer validation completed")
        else:
            raise RuntimeError("Enhancer validation failed")
            
        logger.info("✅ PHASE 2 COMPLETED SUCCESSFULLY")
        return True
        
    def run_phase3(self):
        """Run Phase 3: Integration & reporting."""
        logger.info("=" * 60)
        logger.info("STARTING PHASE 3: INTEGRATION & REPORTING")
        logger.info("=" * 60)
        
        # Step 3.1: Generate final report
        logger.info("Step 3.1: Generating comprehensive final report...")
        args = []
        if self.target_snp:
            args.extend(['--target-snp', self.target_snp])
            
        success, output = self.run_script(
            self.phase3_dir / 'generate_final_report.py',
            args=args
        )
        if success:
            self.workflow_state['phase3_report'] = True
            logger.info("✓ Final report generation completed")
        else:
            raise RuntimeError("Report generation failed")
            
        logger.info("✅ PHASE 3 COMPLETED SUCCESSFULLY")
        return True
        
    def save_workflow_state(self):
        """Save workflow completion state."""
        state_file = self.output_dir / 'workflow_state.json'
        
        workflow_summary = {
            'disease': self.disease_name,
            'target_snp': self.target_snp,
            'workflow_state': self.workflow_state,
            'config_file': str(self.config_file),
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phases_completed': sum(self.workflow_state.values()),
            'total_phases': len(self.workflow_state)
        }
        
        with open(state_file, 'w') as f:
            json.dump(workflow_summary, f, indent=2)
            
        logger.info(f"Workflow state saved to: {state_file}")
        
    def run_complete_workflow(self):
        """Run the complete analysis workflow."""
        logger.info("🚀 STARTING GWAS-ENFORMER-EPIAGENT WORKFLOW")
        logger.info(f"Disease: {self.disease_name}")
        if self.target_snp:
            logger.info(f"Target SNP: {self.target_snp}")
            
        start_time = time.time()
        
        try:
            # Run all phases
            self.run_phase1()
            self.run_phase2() 
            self.run_phase3()
            
            # Save workflow state
            self.save_workflow_state()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {total_time/60:.1f} minutes")
            
            # Print summary
            print("\\n" + "=" * 70)
            print("GWAS-ENFORMER-EPIAGENT WORKFLOW SUMMARY")
            print("=" * 70)
            print(f"Disease studied: {self.disease_name}")
            if self.target_snp:
                print(f"Target SNP: {self.target_snp}")
            print(f"Total execution time: {total_time/60:.1f} minutes")
            print(f"Results saved to: {self.output_dir}")
            print("")
            print("Completed phases:")
            for phase, completed in self.workflow_state.items():
                status = "✅" if completed else "❌"
                print(f"  {status} {phase}")
            print("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            self.save_workflow_state()
            
            # Print failure summary
            print("\\n" + "=" * 70)
            print("WORKFLOW FAILED")
            print("=" * 70)
            print(f"Error: {str(e)}")
            print("")
            print("Completed phases:")
            for phase, completed in self.workflow_state.items():
                status = "✅" if completed else "❌"
                print(f"  {status} {phase}")
            print("=" * 70)
            
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Run complete GWAS-Enformer-EpiAgent workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_workflow.py --config config/config.ini

  # Override disease name
  python run_workflow.py --config config/config.ini --disease "Alzheimer's disease"
  
  # Specify target SNP
  python run_workflow.py --config config/config.ini --target-snp rs7412
  
  # Run specific phases only
  python run_workflow.py --config config/config.ini --phases 1,2
        """
    )
    
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--disease', help='Disease name (overrides config)')
    parser.add_argument('--target-snp', help='Target SNP rsID')
    parser.add_argument('--dataset-id', help='scATAC-seq dataset ID')
    parser.add_argument('--phases', help='Comma-separated phases to run (1,2,3). Default: all')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Validate configuration file
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    try:
        # Initialize orchestrator
        orchestrator = WorkflowOrchestrator(
            args.config, 
            args.disease, 
            args.target_snp, 
            args.dataset_id
        )
        
        if args.dry_run:
            print("DRY RUN MODE - Would execute:")
            print(f"  Disease: {orchestrator.disease_name}")
            print(f"  Configuration: {args.config}")
            if args.target_snp:
                print(f"  Target SNP: {args.target_snp}")
            print("  Phases: All (1, 2, 3)")
            sys.exit(0)
            
        # Determine which phases to run
        if args.phases:
            phases_to_run = [int(p.strip()) for p in args.phases.split(',')]
        else:
            phases_to_run = [1, 2, 3]
            
        # Run specified phases
        success = True
        if 1 in phases_to_run:
            success &= orchestrator.run_phase1()
        if 2 in phases_to_run and success:
            success &= orchestrator.run_phase2()
        if 3 in phases_to_run and success:
            success &= orchestrator.run_phase3()
            
        # Save state and exit
        orchestrator.save_workflow_state()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Workflow initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
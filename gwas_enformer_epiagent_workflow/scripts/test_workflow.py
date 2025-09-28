#!/usr/bin/env python3
"""
Test script for GWAS-Enformer-EpiAgent workflow

This script runs a quick test of the workflow components to ensure
they are properly installed and configured.

Usage:
    python test_workflow.py --config ../config/config.ini
"""

import argparse
import sys
import subprocess
from pathlib import Path
import logging
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_python_imports():
    """Test that all required Python packages can be imported."""
    logger.info("Testing Python package imports...")
    
    required_packages = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
        'sklearn', 'scanpy', 'anndata', 'torch', 'requests',
        'pyfaidx', 'pysam', 'configparser', 'h5py', 'json'
    ]
    
    optional_packages = [
        'tensorflow', 'tensorflow_hub', 'epiagent', 'transformers',
        'episcanpy', 'POT', 'faiss', 'Bio'
    ]
    
    failed_required = []
    failed_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError as e:
            logger.error(f"✗ {package}: {e}")
            failed_required.append(package)
            
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} (optional)")
        except ImportError:
            logger.warning(f"⚠ {package} (optional): not available")
            failed_optional.append(package)
            
    if failed_required:
        logger.error(f"Required packages missing: {', '.join(failed_required)}")
        return False
        
    if failed_optional:
        logger.warning(f"Optional packages missing: {', '.join(failed_optional)}")
        logger.warning("Workflow will run with mock implementations for missing packages")
        
    logger.info("✓ Package import test passed")
    return True


def test_config_file(config_file):
    """Test that the configuration file is valid."""
    logger.info("Testing configuration file...")
    
    if not Path(config_file).exists():
        logger.error(f"Configuration file not found: {config_file}")
        return False
        
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)
        
        required_sections = ['project', 'disease', 'gwas', 'enformer', 'epiagent', 'data', 'analysis']
        
        for section in required_sections:
            if not config.has_section(section):
                logger.error(f"Missing required section: [{section}]")
                return False
            else:
                logger.info(f"✓ Section [{section}] found")
                
        logger.info("✓ Configuration file test passed")
        return True
        
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        return False


def test_workflow_scripts():
    """Test that all workflow scripts exist and are executable."""
    logger.info("Testing workflow scripts...")
    
    workflow_dir = Path(__file__).parent.parent
    
    required_scripts = [
        'phase1_gwas_enformer/gwas_analysis.py',
        'phase1_gwas_enformer/prepare_enformer_sequences.py',
        'phase1_gwas_enformer/run_enformer_predictions.py',
        'phase1_gwas_enformer/identify_functional_enhancer.py',
        'phase2_epiagent_validation/epiagent_preprocessing.py',
        'phase2_epiagent_validation/epiagent_cell_identification.py',
        'phase2_epiagent_validation/validate_enhancer_specificity.py',
        'phase3_integration_reporting/generate_final_report.py',
        'scripts/run_workflow.py'
    ]
    
    missing_scripts = []
    
    for script_path in required_scripts:
        full_path = workflow_dir / script_path
        if full_path.exists():
            logger.info(f"✓ {script_path}")
        else:
            logger.error(f"✗ {script_path} not found")
            missing_scripts.append(script_path)
            
    if missing_scripts:
        logger.error(f"Missing scripts: {', '.join(missing_scripts)}")
        return False
        
    logger.info("✓ Workflow scripts test passed")
    return True


def test_dry_run(config_file):
    """Test dry run of the main workflow."""
    logger.info("Testing workflow dry run...")
    
    try:
        workflow_dir = Path(__file__).parent.parent
        workflow_script = workflow_dir / 'scripts' / 'run_workflow.py'
        
        if not workflow_script.exists():
            logger.error(f"Workflow script not found: {workflow_script}")
            return False
            
        cmd = [
            sys.executable, str(workflow_script),
            '--config', str(config_file),
            '--dry-run'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✓ Workflow dry run passed")
            logger.info(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"Workflow dry run failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Workflow dry run timed out")
        return False
    except Exception as e:
        logger.error(f"Error during dry run: {e}")
        return False


def test_quick_phase1(config_file):
    """Test Phase 1 with minimal data."""
    logger.info("Testing Phase 1 with test data...")
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy config and modify for test
            import configparser
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # Set temporary output directory
            config['data']['output_dir'] = temp_dir
            
            test_config_file = Path(temp_dir) / 'test_config.ini'
            with open(test_config_file, 'w') as f:
                config.write(f)
                
            # Run GWAS analysis only (quickest test)
            workflow_dir = Path(__file__).parent.parent
            gwas_script = workflow_dir / 'phase1_gwas_enformer' / 'gwas_analysis.py'
            
            cmd = [
                sys.executable, str(gwas_script),
                '--config', str(test_config_file),
                '--disease', 'Test Disease'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✓ Phase 1 test passed")
                
                # Check if output files were created
                output_files = list(Path(temp_dir).glob('*'))
                if output_files:
                    logger.info(f"Created {len(output_files)} output files")
                    return True
                else:
                    logger.warning("No output files created, but script completed successfully")
                    return True
            else:
                logger.error(f"Phase 1 test failed: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        logger.error("Phase 1 test timed out")
        return False
    except Exception as e:
        logger.error(f"Error during Phase 1 test: {e}")
        return False


def run_all_tests(config_file):
    """Run all workflow tests."""
    logger.info("=" * 60)
    logger.info("RUNNING GWAS-ENFORMER-EPIAGENT WORKFLOW TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Python Imports", test_python_imports),
        ("Configuration File", lambda: test_config_file(config_file)),
        ("Workflow Scripts", test_workflow_scripts),
        ("Dry Run", lambda: test_dry_run(config_file)),
        ("Phase 1 Quick Test", lambda: test_quick_phase1(config_file))
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\\nRunning test: {test_name}")
        logger.info("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
            
    # Print summary
    logger.info("\\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
            
    logger.info("-" * 60)
    logger.info(f"PASSED: {passed}/{total} tests")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Workflow is ready to use!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed - Please check the issues above")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test GWAS-Enformer-EpiAgent workflow')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--quick', action='store_true', help='Run only quick tests (skip Phase 1 test)')
    
    args = parser.parse_args()
    
    if args.quick:
        logger.info("Running quick tests only...")
        
        tests = [
            ("Python Imports", test_python_imports),
            ("Configuration File", lambda: test_config_file(args.config)),
            ("Workflow Scripts", test_workflow_scripts),
            ("Dry Run", lambda: test_dry_run(args.config))
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            logger.info(f"\\nRunning test: {test_name}")
            try:
                result = test_func()
                all_passed &= result
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                all_passed = False
                
        if all_passed:
            logger.info("✅ All quick tests passed!")
        else:
            logger.error("❌ Some quick tests failed")
            
        sys.exit(0 if all_passed else 1)
        
    else:
        success = run_all_tests(args.config)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
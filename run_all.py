"""
Master script to run entire NLSE simulation pipeline

This script executes:
1. Parameter validation
2. All three solvers (linear, nonlinear, full NLSE)
3. Visualization of results
4. Comparison plots
"""

import sys
import logging

# Configure logging once for the entire pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def run_tests():
    logger.info("="*60)
    logger.info("STEP 0: Running validation tests")
    logger.info("="*60)
    try:
        import test_suite
        success = test_suite.run_all_tests()
        if not success:
            logger.error("Tests failed. Please fix errors before continuing.")
            return False
        logger.info("All tests passed!\n")
        return True
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def validate_parameters():
    logger.info("="*60)
    logger.info("STEP 1: Parameter Validation")
    logger.info("="*60)
    try:
        import parameters
        parameters.print_parameters()
        parameters.validate_parameters()
        return True
    except Exception as e:
        logger.error(f"Error in parameter validation: {e}")
        return False

def run_solvers():
    solvers = [
        ("linear_solver", "Linear Solver"),
        ("nonlinear_solver", "Nonlinear Solver"),
        ("nlse_solver", "Full NLSE Solver")
    ]
    
    for module_name, display_name in solvers:
        logger.info("="*60)
        logger.info(f"STEP 2: Running {display_name}")
        logger.info("="*60)
        try:
            module = __import__(module_name)
            module.main()
            logger.info(f"{display_name} completed successfully.\n")
        except Exception as e:
            logger.error(f"Error running {display_name}: {e}")
            return False
    
    return True

def create_visualizations():
    logger.info("="*60)
    logger.info("STEP 3: Creating Visualizations")
    logger.info("="*60)
    try:
        import visualize
        visualize.main()
        logger.info("Visualizations completed.\n")
        return True
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return False

def create_comparisons():
    logger.info("=" * 60)
    logger.info("STEP 4: Creating Comparison Plots")
    logger.info("=" * 60)
    try:
        import compare_results
        logger.info("Generating comparison plots...")
        compare_results.main()   # clean, isolated, no exec garbage
        logger.info("Comparison plots completed.\n")
        return True
    except FileNotFoundError:
        logger.warning("compare_results.py not found, skipping comparisons.")
        return True
    except Exception as e:
        logger.error(f"Error creating comparisons: {e}")
        return False

def main():
    skip_tests = "--skip-tests" in sys.argv
    
    logger.info("*"*60)
    logger.info("NLSE Simulation Pipeline")
    logger.info("*"*60)
    logger.info("")
    
    # Step 0: Run tests (optional)
    if not skip_tests:
        if not run_tests():
            logger.error("Pipeline aborted due to test failures.")
            return False
    else:
        logger.info("Skipping tests (--skip-tests flag detected)\n")
    
    # Step 1: Validate parameters
    if not validate_parameters():
        logger.error("Pipeline aborted due to parameter validation errors.")
        return False
    
    # Step 2: Run all solvers
    if not run_solvers():
        logger.error("Pipeline aborted due to solver errors.")
        return False
    
    # Step 3: Create visualizations
    if not create_visualizations():
        logger.warning("Visualization step failed, but continuing...")
    
    # Step 4: Create comparisons
    if not create_comparisons():
        logger.warning("Comparison step failed, but continuing...")
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info("Results saved to:")
    logger.info("  - *_solver_results.npz (simulation data)")
    logger.info("  - plots/ (visualizations)")
    logger.info("")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
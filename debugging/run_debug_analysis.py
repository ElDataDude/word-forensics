"""Script to run document analysis with debug logging and analysis."""

import os
import logging
from pathlib import Path
from debug_handler import setup_debug_logging
from forensic_analysis import WordForensicAnalyzer
from statistical import ForensicStatisticalAnalyzer
import json

def run_analysis_with_debug():
    """Run document analysis with debug logging enabled."""
    # Set up debug logging
    debug_handler = setup_debug_logging()
    
    try:
        # Initialize analyzers with actual paths
        base_dir = Path(os.getcwd())
        input_dir = base_dir / "input"
        target_dir = input_dir / "target"
        same_origin_dir = input_dir / "same_origin"
        reference_dir = input_dir / "reference"
        output_dir = base_dir / "output"
        cache_dir = base_dir / "cache"
        
        # Log directory setup
        logging.info(f"Working directory: {base_dir}")
        logging.info(f"Input directory: {input_dir}")
        logging.info(f"Target directory: {target_dir}")
        logging.info(f"Same-origin directory: {same_origin_dir}")
        logging.info(f"Reference directory: {reference_dir}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Cache directory: {cache_dir}")
        
        # Ensure directories exist
        for dir_path in [reference_dir, output_dir, cache_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
            logging.debug(f"Created/verified directory: {dir_path}")
        
        # Find target document
        target_docs = list(target_dir.glob("*.docx"))
        if not target_docs:
            logging.error("No target .docx file found in target directory")
            return
        target_doc = target_docs[0]
        logging.info(f"Found target document: {target_doc}")
            
        # Find same-origin document
        same_origin_docs = list(same_origin_dir.glob("*.docx"))
        if not same_origin_docs:
            logging.error("No same-origin .docx file found in same-origin directory")
            return
        same_origin_doc = same_origin_docs[0]
        logging.info(f"Found same-origin document: {same_origin_doc}")
        
        # Check reference documents
        reference_docs = list(reference_dir.glob("*.docx"))
        if not reference_docs:
            logging.error("No reference .docx files found in reference directory")
            return
        logging.info(f"Found {len(reference_docs)} reference documents")
            
        # Initialize analyzers
        logging.info("Initializing forensic analyzer...")
        try:
            forensic_analyzer = WordForensicAnalyzer(
                target_path=str(target_doc),
                same_origin_path=str(same_origin_doc),
                reference_files=[str(file) for file in reference_docs]
            )
            logging.debug("Forensic analyzer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize forensic analyzer: {e}", exc_info=True)
            raise
            
        logging.info("Initializing statistical analyzer...")
        try:
            statistical_analyzer = ForensicStatisticalAnalyzer(reference_dir, cache_dir)
            statistical_analyzer.set_analyzer(forensic_analyzer)
            logging.debug("Statistical analyzer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize statistical analyzer: {e}", exc_info=True)
            raise
        
        logging.info("Both analyzers initialized successfully")
        
        # Run the analysis
        try:
            logging.info("Starting file comparison analysis...")
            comparison_result = forensic_analyzer.compare_files()
            
            if not comparison_result:
                logging.error("File comparison analysis returned no results")
                return
                
            logging.info("File comparison completed successfully")
            logging.debug(f"Comparison result keys: {list(comparison_result.keys())}")
            
            # Generate report and summary
            report_result = forensic_analyzer.generate_report()
            
            # Extract report and summary
            report = report_result
            summary = report_result.get("summary", "")
            
            # Save report to file
            report_file = output_dir / "analysis_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            logging.debug(f"Report saved to {report_file}")
            
            # Save summary to file
            summary_file = output_dir / "analysis_summary.txt"
            with open(summary_file, "w") as f:
                f.write(summary)
            logging.debug(f"Summary saved to {summary_file}")
            
            return True
        
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}", exc_info=True)
            
    except Exception as e:
        logging.error(f"Setup failed: {str(e)}", exc_info=True)
    
    finally:
        # Analyze logs and generate report
        logging.info("Generating debug analysis report...")
        debug_handler.analyze_logs()

if __name__ == "__main__":
    run_analysis_with_debug()

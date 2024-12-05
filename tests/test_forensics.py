"""
Test script demonstrating comprehensive document forensics analysis.
"""

import os
from pathlib import Path
import logging
from pprint import pformat
from statistical import DocumentForensicsAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize paths
    reference_dir = Path('input/reference')
    target_dir = Path('input/target')
    same_origin_dir = Path('input/same_origin')
    
    # Get reference files
    reference_files = list(reference_dir.glob('*.docx'))
    
    if not reference_files:
        logger.error("No reference files found. Please add .docx files to the input/reference directory.")
        return
        
    # Initialize analyzer
    analyzer = DocumentForensicsAnalyzer()
    
    # Analyze reference set
    logger.info("\nAnalyzing reference set...")
    ref_stats = analyzer.analyze_reference_set([str(f) for f in reference_files])
    
    logger.info("Reference set statistics:")
    logger.info(f"Total pairs analyzed: {ref_stats['total_pairs']}")
    logger.info(f"Never matching fields: {ref_stats['never_matching']}")
    logger.info(f"Always matching fields: {ref_stats['always_matching']}")
    logger.info(f"Field frequencies: {pformat(ref_stats['field_frequencies'])}")
    
    # Get target and same_origin files
    target_files = list(target_dir.glob('*.docx'))
    same_origin_files = list(same_origin_dir.glob('*.docx'))
    
    if not target_files or not same_origin_files:
        logger.error("Need both target and same_origin files for comparison")
        return
        
    # Analyze target vs same_origin file
    target_file = target_files[0]
    same_origin_file = same_origin_files[0]
    
    logger.info("\nAnalyzing file pair:")
    logger.info(f"Target: {target_file.name}")
    logger.info(f"Same Origin: {same_origin_file.name}")
    
    # Run analysis
    results = analyzer.analyze_target_pair(str(target_file), str(same_origin_file))
    
    # Output results
    logger.info("\nAnalysis Results:")
    
    # Print the overall assessment based on confidence score
    confidence_score = results['confidence_score']
    if confidence_score > 0.8:
        assessment = "HIGH probability of shared origin"
    elif confidence_score > 0.5:
        assessment = "MODERATE probability of shared origin"
    else:
        assessment = "LOW probability of shared origin"
    
    logger.info(f"Overall Assessment: {assessment}")
    logger.info(f"Confidence Score: {confidence_score:.2f}\n")
    
    # Print forensic analysis results by category
    logger.info("Forensic Analysis Results:")
    
    categories = {
        'Core Metadata': [f for f in results['matching_fields'] if f.startswith('metadata_')],
        'Content Structure': [f for f in results['matching_fields'] if f.startswith('structure_')],
        'OOXML Analysis': [f for f in results['matching_fields'] if f.startswith('ooxml_')],
        'Binary Analysis': [f for f in results['matching_fields'] if f.startswith('binary_')]
    }
    
    for category, fields in categories.items():
        if fields:
            logger.info(f"\n{category}:")
            for field in fields:
                # Check if this field is rare in the reference set
                freq = ref_stats['field_frequencies'].get(field, 0)
                if freq < 0.1 and field not in ref_stats['always_matching']:
                    logger.info(f"- {field} (Significant: only matches in {freq:.1%} of reference pairs)")
                else:
                    logger.info(f"- {field}")
    
    # Print any anomalies
    if results['anomalies']:
        logger.info("\nAnomalous Matches:")
        for anomaly in results['anomalies']:
            logger.info(f"- {anomaly['field']}: {anomaly['value']} ({anomaly['reason']})")
    
    # Print any suspicious matches
    if results['suspicious_matches']:
        logger.info("\nSuspicious Matches:")
        for match in results['suspicious_matches']:
            logger.info(f"- {match['field']}: {match['value']} "
                      f"(occurs in {match['frequency']:.1%} of reference pairs)")

if __name__ == '__main__':
    main()

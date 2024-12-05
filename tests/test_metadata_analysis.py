"""
Test script demonstrating the usage of MetadataPairwiseAnalyzer.
"""

import os
from pathlib import Path
import logging
from docx import Document
from statistical import MetadataPairwiseAnalyzer
from itertools import combinations
from pprint import pformat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_get_property(props, prop_name, default=''):
    """Safely get a property value, returning default if not found."""
    try:
        value = getattr(props, prop_name, default)
        return str(value) if value is not None else default
    except:
        return default

def extract_metadata(docx_path):
    """Extract metadata from a Word document."""
    doc = Document(docx_path)
    props = doc.core_properties
    
    return {
        'creator': safe_get_property(props, 'author'),
        'last_modified_by': safe_get_property(props, 'last_modified_by'),
        'company': safe_get_property(props, 'company'),
        'revision_number': safe_get_property(props, 'revision'),
        'template': safe_get_property(doc.settings, 'template'),
        'application': 'Microsoft Office Word',  # Usually constant
        'application_version': '16.0',  # Example version
        'total_editing_time': safe_get_property(props, 'total_editing_time', '0')
    }

def find_matching_pairs(files_with_metadata, field):
    """Find pairs of files with matching metadata field values."""
    matching_pairs = []
    for (file1, meta1), (file2, meta2) in combinations(files_with_metadata, 2):
        if meta1[field] and meta2[field] and meta1[field] == meta2[field]:
            # Only count non-empty values
            if meta1[field].strip():
                matching_pairs.append((file1, file2, meta1[field]))
    return matching_pairs

def main():
    # Initialize paths
    reference_dir = Path('input/reference')
    target_dir = Path('input/target')
    same_origin_dir = Path('input/same_origin')
    
    # Get reference files with metadata
    reference_files_with_metadata = []
    for file in reference_dir.glob('*.docx'):
        try:
            metadata = extract_metadata(file)
            reference_files_with_metadata.append((file.name, metadata))
            logger.info(f"Extracted metadata from reference file: {file.name}")
            logger.info(f"Metadata:\n{pformat(metadata)}\n")
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    
    if not reference_files_with_metadata:
        logger.error("No reference files found. Please add .docx files to the input/reference directory.")
        return
    
    # Find matching pairs in reference set
    logger.info("\nAnalyzing reference set for matching pairs...")
    
    # Check creator matches between different files
    creator_matches = find_matching_pairs(reference_files_with_metadata, 'creator')
    if creator_matches:
        logger.info("\nDifferent files with matching creators:")
        for file1, file2, creator in creator_matches:
            logger.info(f"- {file1}")
            logger.info(f"  {file2}")
            logger.info(f"  Creator: {creator}\n")
    
    # Check last_modified_by matches between different files
    modified_matches = find_matching_pairs(reference_files_with_metadata, 'last_modified_by')
    if modified_matches:
        logger.info("\nDifferent files with matching last_modified_by:")
        for file1, file2, modifier in modified_matches:
            logger.info(f"- {file1}")
            logger.info(f"  {file2}")
            logger.info(f"  Modified by: {modifier}\n")
    
    # Initialize analyzer
    analyzer = MetadataPairwiseAnalyzer()
    
    # Convert to list of just metadata for analyzer
    reference_metadata = [meta for _, meta in reference_files_with_metadata]
    
    # Analyze reference set
    logger.info("\nAnalyzing reference set statistics...")
    ref_stats = analyzer.analyze_reference_set(reference_metadata)
    logger.info("Reference set statistics:")
    logger.info(f"Total pairs analyzed: {ref_stats['total_pairs']}")
    logger.info(f"Never matching fields: {ref_stats['never_matching']}")
    logger.info(f"Always matching fields: {ref_stats['always_matching']}")
    logger.info(f"Field frequencies: {ref_stats['field_frequencies']}")
    
    # Get target and same_origin files
    target_files = list(target_dir.glob('*.docx'))
    same_origin_files = list(same_origin_dir.glob('*.docx'))
    
    if not target_files or not same_origin_files:
        logger.error("Need both target and same_origin files for comparison")
        return
    
    # Analyze target vs same_origin file
    target_file = target_files[0]
    same_origin_file = same_origin_files[0]
    logger.info(f"\nAnalyzing file pair:")
    logger.info(f"Target: {target_file.name}")
    logger.info(f"Same Origin: {same_origin_file.name}")
    
    try:
        target_metadata = extract_metadata(target_file)
        same_origin_metadata = extract_metadata(same_origin_file)
        
        logger.info(f"\nTarget file metadata:\n{pformat(target_metadata)}")
        logger.info(f"\nSame origin file metadata:\n{pformat(same_origin_metadata)}")
        
        results = analyzer.analyze_target_pair(target_metadata, same_origin_metadata)
        
        # Print analysis results
        logger.info("\nAnalysis Results:")
        logger.info(results['analysis_summary'])
        
        if results['anomalies']:
            logger.info("\nDetailed Anomalies:")
            for anomaly in results['anomalies']:
                logger.info(f"Field: {anomaly['field']}")
                logger.info(f"Value: {anomaly['value']}")
                logger.info(f"Reason: {anomaly['reason']}")
        
        if results['suspicious_matches']:
            logger.info("\nSuspicious Matches:")
            for match in results['suspicious_matches']:
                logger.info(f"Field: {match['field']}")
                logger.info(f"Value: {match['value']}")
                logger.info(f"Frequency in reference set: {match['frequency']:.1%}")
        
    except Exception as e:
        logger.error(f"Error analyzing files: {e}")

if __name__ == '__main__':
    main()

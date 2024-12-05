"""
Metadata Pairwise Analysis Module

This module implements statistical analysis for detecting anomalous metadata patterns
between pairs of Word documents. It compares metadata patterns between target files
against a reference set to identify unusual matches that might indicate shared origin.
"""

from itertools import combinations
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MetadataPairwiseAnalyzer:
    """Analyzes metadata patterns between pairs of documents to detect anomalies."""

    def __init__(self, metadata_fields: List[str] = None):
        """
        Initialize the analyzer with specific metadata fields to compare.
        
        Args:
            metadata_fields: List of metadata field names to analyze. If None,
                           will use a default set of fields.
        """
        self.metadata_fields = metadata_fields or [
            # Core Metadata
            'creator',
            'last_modified_by',
            'company',
            'template',
            'application',
            'application_version',
            'revision_number',
            'total_editing_time',
            # Content Structure
            'structure_sections',
            'structure_tables',
            # OOXML Analysis
            'ooxml_content_types',
            'ooxml_namespaces',
            'ooxml_relationships',
            # Binary Analysis
            'binary_signature',
            'binary_ole_streams'
        ]
        
        # Statistics from reference set analysis
        self.field_match_frequencies = defaultdict(float)  # Field -> frequency of matches
        self.never_matching_fields = set()  # Fields that never match in reference set
        self.always_matching_fields = set()  # Fields that always match in reference set
        self.field_patterns = defaultdict(set)  # Field -> set of observed values
        
        self.reference_analyzed = False

    def analyze_reference_set(self, reference_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the reference set to establish baseline patterns.
        
        Args:
            reference_files: List of dictionaries containing metadata from reference files
            
        Returns:
            Dictionary containing reference set statistics
        """
        if not reference_files:
            raise ValueError("Reference set cannot be empty")
            
        total_pairs = 0
        matches_by_field = defaultdict(int)
        
        # Analyze all pairs of reference files
        for file1, file2 in combinations(reference_files, 2):
            total_pairs += 1
            
            # Count matching fields
            for field in self.metadata_fields:
                if field not in file1 or field not in file2:
                    continue
                    
                # Record observed values
                self.field_patterns[field].add(str(file1[field]))
                self.field_patterns[field].add(str(file2[field]))
                
                if file1[field] == file2[field]:
                    matches_by_field[field] += 1
        
        # Calculate match frequencies
        for field in self.metadata_fields:
            freq = matches_by_field[field] / total_pairs if total_pairs > 0 else 0
            self.field_match_frequencies[field] = freq
            
            if freq == 0:
                self.never_matching_fields.add(field)
            elif freq == 1:
                self.always_matching_fields.add(field)
        
        self.reference_analyzed = True
        
        return {
            'total_pairs': total_pairs,
            'field_frequencies': dict(self.field_match_frequencies),
            'never_matching': list(self.never_matching_fields),
            'always_matching': list(self.always_matching_fields),
            'unique_patterns': {k: len(v) for k, v in self.field_patterns.items()}
        }

    def analyze_target_pair(self, file1: Dict[str, Any], 
                          file2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a pair of target files for anomalous metadata matches.
        
        Args:
            file1: Metadata dictionary from first target file
            file2: Metadata dictionary from second target file
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.reference_analyzed:
            raise RuntimeError("Must analyze reference set before analyzing target pair")

        anomalies = []
        suspicious_matches = []
        matching_fields = []
        
        # Analyze each metadata field
        for field in self.metadata_fields:
            if field not in file1 or field not in file2:
                continue
                
            if file1[field] == file2[field]:
                matching_fields.append(field)
                
                # Check if this is an anomaly
                if field in self.never_matching_fields:
                    anomalies.append({
                        'field': field,
                        'value': file1[field],
                        'reason': 'Never matches in reference set'
                    })
                
                # Check if this is a suspicious match
                elif (self.field_match_frequencies[field] < 0.1 and 
                      field not in self.always_matching_fields):
                    suspicious_matches.append({
                        'field': field,
                        'value': file1[field],
                        'frequency': self.field_match_frequencies[field]
                    })
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            matching_fields, anomalies, suspicious_matches
        )
        
        return {
            'matching_fields': matching_fields,
            'anomalies': anomalies,
            'suspicious_matches': suspicious_matches,
            'confidence_score': confidence_score,
            'analysis_summary': self._generate_summary(
                matching_fields, anomalies, suspicious_matches, confidence_score
            )
        }

    def _calculate_confidence_score(self, matching_fields: List[str],
                                 anomalies: List[Dict], 
                                 suspicious_matches: List[Dict]) -> float:
        """
        Calculate a confidence score for shared origin hypothesis.
        
        Args:
            matching_fields: List of fields that match between target files
            anomalies: List of anomalous matches
            suspicious_matches: List of suspicious matches
            
        Returns:
            Confidence score between 0 and 1
        """
        if not matching_fields:
            return 0.0

        # Base score from number of matching fields
        base_score = len(matching_fields) / len(self.metadata_fields)
        
        # Adjust score based on anomalies and suspicious matches
        anomaly_bonus = len(anomalies) * 0.15  # Anomalous matches increase confidence
        suspicious_bonus = sum(0.1 for _ in suspicious_matches)  # Suspicious matches increase confidence
        
        # Calculate final score
        score = base_score + anomaly_bonus + suspicious_bonus
        score = min(max(score, 0.0), 1.0)
        
        return score

    def _generate_summary(self, matching_fields: List[str],
                        anomalies: List[Dict],
                        suspicious_matches: List[Dict],
                        confidence_score: float) -> str:
        """
        Generate a human-readable summary of the analysis results.
        
        Args:
            matching_fields: List of fields that match between target files
            anomalies: List of anomalous matches
            suspicious_matches: List of suspicious matches
            confidence_score: Calculated confidence score
            
        Returns:
            String containing analysis summary
        """
        summary_parts = []
        
        # Overall assessment
        if confidence_score > 0.8:
            assessment = "HIGH probability of shared origin"
        elif confidence_score > 0.5:
            assessment = "MODERATE probability of shared origin"
        else:
            assessment = "LOW probability of shared origin"
            
        summary_parts.append(f"Overall Assessment: {assessment}")
        summary_parts.append(f"Confidence Score: {confidence_score:.2f}\n")
        
        # Group matching fields by category
        categories = {
            'Core Metadata': ['creator', 'last_modified_by', 'company', 'template', 
                            'application', 'application_version', 'revision_number', 
                            'total_editing_time'],
            'Content Structure': ['structure_sections', 'structure_tables'],
            'OOXML Analysis': ['ooxml_content_types', 'ooxml_namespaces', 'ooxml_relationships'],
            'Binary Analysis': ['binary_signature', 'binary_ole_streams']
        }
        
        if matching_fields:
            summary_parts.append("Forensic Analysis Results:")
            for category, fields in categories.items():
                matching_in_category = [f for f in matching_fields if f in fields]
                if matching_in_category:
                    summary_parts.append(f"\n{category}:")
                    for field in matching_in_category:
                        summary_parts.append(f"- {field}")
        
        # Anomalies
        if anomalies:
            summary_parts.append("\nAnomalous Matches:")
            for anomaly in anomalies:
                summary_parts.append(
                    f"- {anomaly['field']}: {anomaly['value']} "
                    f"({anomaly['reason']})"
                )
        
        # Suspicious matches
        if suspicious_matches:
            summary_parts.append("\nSuspicious Matches:")
            for match in suspicious_matches:
                summary_parts.append(
                    f"- {match['field']}: {match['value']} "
                    f"(occurs in {match['frequency']:.1%} of reference pairs)"
                )
        
        return "\n".join(summary_parts)

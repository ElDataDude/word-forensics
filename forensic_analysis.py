#!/usr/bin/env python3

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import binascii
import difflib
from openai import OpenAI
import logging
import time
import hashlib
from docx import Document
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from statistical import ForensicStatisticalAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format for terminal
    handlers=[
        logging.StreamHandler(),  # Terminal output
        logging.FileHandler('forensics.log')  # File output with full debug info
    ]
)

# Set up file logger with detailed format
file_handler = logging.FileHandler('forensics.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up console handler with minimal format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Remove default handlers and add our custom ones
logger = logging.getLogger()
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class WordForensicAnalyzer:
    def __init__(self, target_path: str, same_origin_path: str, reference_files: List[str]):
        """Initialize the analyzer with paths to documents for comparison."""
        load_dotenv()
        
        # Initialize OpenAI client with minimal configuration
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",  # Explicitly set the base URL
            timeout=60.0,  # Set a reasonable timeout
            max_retries=3  # Set max retries
        )
        
        self.target_path = Path(target_path)
        self.same_origin_path = Path(same_origin_path)
        self.reference_files = [Path(file) for file in reference_files]
        
        # Create output directory within project structure
        self.output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache directory for storing intermediate results
        self.cache_dir = self.output_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize statistical analyzer
        self.statistical_analyzer = ForensicStatisticalAnalyzer(
            reference_dir=Path(os.path.dirname(os.path.abspath(__file__))) / "input" / "reference",  # Use the parent directory of the reference files
            cache_dir=self.cache_dir
        )
        self.statistical_analyzer.set_analyzer(self)
        
    def validate_files(self):
        """Validate that all input files exist and are .docx files."""
        files_to_check = []
        
        # Add target and same-origin files if they exist
        if self.target_path:
            files_to_check.append(self.target_path)
        if self.same_origin_path:
            files_to_check.append(self.same_origin_path)
            
        # Add reference files if they exist
        if self.reference_files:
            files_to_check.extend(self.reference_files)
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            if not str(file_path).lower().endswith('.docx'):
                raise ValueError(f"Invalid file format for {file_path}. Must be .docx")
        
    def extract_metadata(self, docx_path: Path) -> Dict[str, Any]:
        """Extract metadata from a Word document."""
        try:
            doc = Document(docx_path)
            if not hasattr(doc, 'core_properties'):
                logging.warning(f"No core properties found in {docx_path.name}")
                return {}
                
            core_props = doc.core_properties
            
            # Helper function to safely get attributes
            def get_prop(obj, attr):
                try:
                    val = getattr(obj, attr, None)
                    # Convert datetime objects to string
                    if hasattr(val, 'isoformat'):
                        return val.isoformat()
                    return val
                except Exception as e:
                    logging.warning(f"Error getting property {attr}: {str(e)}")
                    return None
            
            metadata = {
                "author": get_prop(core_props, 'author'),
                "last_modified_by": get_prop(core_props, 'last_modified_by'),
                "created": get_prop(core_props, 'created'),
                "modified": get_prop(core_props, 'modified'),
                "revision": get_prop(core_props, 'revision'),
                "title": get_prop(core_props, 'title'),
                "category": get_prop(core_props, 'category'),
                "keywords": get_prop(core_props, 'keywords'),
                "comments": get_prop(core_props, 'comments'),
                "identifier": get_prop(core_props, 'identifier'),
                "language": get_prop(core_props, 'language'),
                "subject": get_prop(core_props, 'subject'),
                "version": get_prop(core_props, 'version')
            }
            
            # Remove None values for cleaner output
            return {k: v for k, v in metadata.items() if v is not None}
            
        except Exception as e:
            logging.error(f"Error extracting metadata from {docx_path.name}: {str(e)}")
            return {}

    def analyze_content(self, docx_path: Path) -> Dict[str, Any]:
        """Analyze document content including text, styles, and structure."""
        try:
            doc = Document(docx_path)
            
            content_analysis = {
                "paragraphs": 0,
                "sections": 0,
                "styles": [],
                "text_content": "",
                "fonts_used": set(),
                "tracked_changes": [],
                "comments": [],
                "embedded_objects": []
            }
            
            # Safely get document structure
            try:
                content_analysis["paragraphs"] = len(doc.paragraphs)
            except:
                logging.warning(f"Could not count paragraphs in {docx_path.name}")
                
            try:
                content_analysis["sections"] = len(doc.sections)
            except:
                logging.warning(f"Could not count sections in {docx_path.name}")
            
            # Analyze text and styles
            for paragraph in doc.paragraphs:
                try:
                    content_analysis["text_content"] += paragraph.text + "\n"
                    if hasattr(paragraph, 'style') and paragraph.style and paragraph.style.name:
                        if paragraph.style.name not in content_analysis["styles"]:
                            content_analysis["styles"].append(paragraph.style.name)
                    
                    # Extract font information from runs
                    for run in paragraph.runs:
                        if hasattr(run, 'font') and hasattr(run.font, 'name') and run.font.name:
                            content_analysis["fonts_used"].add(run.font.name)
                except Exception as e:
                    logging.warning(f"Error processing paragraph in {docx_path.name}: {str(e)}")
                    continue
            
            # Convert set to list for JSON serialization
            content_analysis["fonts_used"] = list(content_analysis["fonts_used"])
            return content_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing content of {docx_path.name}: {str(e)}")
            return {
                "paragraphs": 0,
                "sections": 0,
                "styles": [],
                "text_content": "",
                "fonts_used": [],
                "tracked_changes": [],
                "comments": [],
                "embedded_objects": [],
                "error": str(e)
            }

    def analyze_ooxml_structure(self, docx_path: Path) -> Dict[str, Any]:
        """Analyze the internal OOXML structure of the document."""
        structure_info = {
            "xml_files": [],
            "relationships": [],
            "content_types": [],
            "element_counts": {},
            "custom_properties": [],
            "error": None
        }
        
        try:
            with zipfile.ZipFile(docx_path, 'r') as zip_ref:
                # Analyze XML files
                for item in zip_ref.namelist():
                    if item.endswith('.xml') or item.endswith('.rels'):
                        try:
                            with zip_ref.open(item) as xml_file:
                                tree = ET.parse(xml_file)
                                root = tree.getroot()
                                
                                structure_info["xml_files"].append({
                                    "path": item,
                                    "root_tag": root.tag,
                                    "namespace": root.tag.split('}')[0].strip('{') if '}' in root.tag else None,
                                    "child_count": len(list(root))
                                })
                                
                                # Count elements
                                element_counts = {}
                                for elem in root.iter():
                                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                                    element_counts[tag] = element_counts.get(tag, 0) + 1
                                structure_info["element_counts"][item] = element_counts
                        except Exception as e:
                            logging.warning(f"Error parsing XML file {item} in {docx_path.name}: {str(e)}")
                            continue
                
                # Analyze relationships
                if '[Content_Types].xml' in zip_ref.namelist():
                    try:
                        with zip_ref.open('[Content_Types].xml') as types_xml:
                            types_tree = ET.parse(types_xml)
                            types_root = types_tree.getroot()
                            for override in types_root.findall('{http://schemas.openxmlformats.org/package/2006/content-types}Override'):
                                structure_info["content_types"].append({
                                    "part_name": override.get('PartName'),
                                    "content_type": override.get('ContentType')
                                })
                    except Exception as e:
                        logging.warning(f"Error parsing content types in {docx_path.name}: {str(e)}")
                
                # Analyze custom properties
                if 'docProps/custom.xml' in zip_ref.namelist():
                    try:
                        with zip_ref.open('docProps/custom.xml') as custom_xml:
                            custom_tree = ET.parse(custom_xml)
                            custom_root = custom_tree.getroot()
                            for prop in custom_root.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/custom-properties}property'):
                                structure_info["custom_properties"].append({
                                    "name": prop.get('name'),
                                    "type": next(iter(prop)).tag.split('}')[-1] if len(prop) > 0 else None,
                                    "value": next(iter(prop)).text if len(prop) > 0 else None
                                })
                    except Exception as e:
                        logging.warning(f"Error parsing custom properties in {docx_path.name}: {str(e)}")
            
            return structure_info
            
        except Exception as e:
            error_msg = f"Error analyzing OOXML structure of {docx_path.name}: {str(e)}"
            logging.error(error_msg)
            structure_info["error"] = error_msg
            return structure_info

    def analyze_document(self, docx_path: Path) -> Dict[str, Any]:
        """
        Analyze a single document by running all available analysis methods.
        Returns a dictionary containing all analysis results.
        """
        try:
            analysis_results = {
                "metadata": self.extract_metadata(docx_path),
                "content": self.analyze_content(docx_path),
                "ooxml": self.analyze_ooxml_structure(docx_path),
                "binary": self.analyze_binary_content(docx_path),
                "file_info": {
                    "name": docx_path.name,
                    "size": docx_path.stat().st_size,
                    "created": datetime.fromtimestamp(docx_path.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(docx_path.stat().st_mtime).isoformat()
                }
            }
            return analysis_results
        except Exception as e:
            logging.error(f"Error analyzing document {docx_path}: {str(e)}")
            return {
                "error": str(e),
                "file_path": str(docx_path)
            }

    def analyze_binary_content(self, docx_path: Path) -> Dict[str, Any]:
        """Perform binary analysis of the document."""
        binary_info = {
            "file_size": 0,
            "readable_strings": [],
            "user_paths": [],
            "template_paths": [],
            "system_markers": [],
            "binary_signatures": [],
            "error": None
        }
        
        try:
            # Get file size
            binary_info["file_size"] = os.path.getsize(docx_path)
            
            # Read file in binary mode
            with open(docx_path, 'rb') as f:
                content = f.read()
                
                # Convert to hex for signature analysis
                hex_content = binascii.hexlify(content[:4096]).decode('utf-8')  # Analyze first 4KB for signatures
                
                # Look for common binary signatures
                signatures = {
                    '504B0304': 'ZIP Archive (DOCX)',
                    '504B0506': 'ZIP End of Central Directory',
                    '504B0708': 'ZIP Data Descriptor'
                }
                
                for sig, desc in signatures.items():
                    if sig.lower() in hex_content.lower():
                        binary_info["binary_signatures"].append({
                            "signature": sig,
                            "description": desc,
                            "offset": hex_content.lower().index(sig.lower()) // 2
                        })
                
                # Extract readable strings (at least 4 chars long)
                current_string = ""
                for byte in content:
                    char = chr(byte)
                    if 32 <= byte <= 126:  # Printable ASCII
                        current_string += char
                    else:
                        if len(current_string) >= 4:
                            # Filter out common noise
                            if not any(noise in current_string.lower() for noise in ['xmlns', 'http://', 'https://']):
                                s = current_string
                                s_lower = s.lower()
                                
                                # User directory paths
                                if '/users/' in s_lower or r'c:\users' in s_lower:
                                    binary_info["user_paths"].append(s)
                                
                                # Template paths
                                elif 'template' in s_lower and ('.dot' in s_lower or '.dotx' in s_lower):
                                    binary_info["template_paths"].append(s)
                                
                                # System markers
                                elif any(marker in s_lower for marker in ['windows', 'microsoft', 'office', 'word']):
                                    binary_info["system_markers"].append(s)
                                
                                # Other readable strings
                                else:
                                    binary_info["readable_strings"].append(s)
                        
                        current_string = ""
            
            # Limit the number of strings to avoid excessive output
            max_strings = 100
            binary_info["readable_strings"] = binary_info["readable_strings"][:max_strings]
            
            return binary_info
            
        except Exception as e:
            error_msg = f"Error analyzing binary content of {docx_path.name}: {str(e)}"
            logging.error(error_msg)
            binary_info["error"] = error_msg
            return binary_info

    def find_origin_evidence(self) -> Dict[str, Any]:
        """Find definitive evidence of shared origin between target and same-origin files."""
        target_binary = self.analyze_binary_content(self.target_path)
        same_origin_binary = self.analyze_binary_content(self.same_origin_path)
        
        target_meta = self.extract_metadata(self.target_path)
        same_origin_meta = self.extract_metadata(self.same_origin_path)
        
        target_content = self.analyze_content(self.target_path)
        same_origin_content = self.analyze_content(self.same_origin_path)
        
        evidence = {
            "definitive_markers": [],
            "strong_indicators": [],
            "potential_indicators": [],
            "metadata_matches": {},
            "binary_signatures": [],
            "shared_paths": {
                "user_paths": [],
                "template_paths": [],
                "system_markers": []
            }
        }
        
        # Helper function to find matching paths
        def find_matching_paths(list1, list2):
            shared = set()
            for path1 in list1:
                for path2 in list2:
                    # If paths match
                    if path1.lower() == path2.lower():
                        shared.add(path1)
            return list(shared)
        
        # Helper function to check metadata matches
        def check_metadata_match(field: str, target_val, same_origin_val) -> bool:
            if not target_val or not same_origin_val:
                return False
            return str(target_val).lower() == str(same_origin_val).lower()
        
        # Check metadata matches
        metadata_fields = {
            "author": "Author name",
            "last_modified_by": "Last modified by",
            "title": "Document title",
            "template": "Template name",
            "company": "Company name",
            "application": "Application name",
            "app_version": "Application version"
        }
        
        for field, description in metadata_fields.items():
            if check_metadata_match(
                field,
                target_meta.get(field),
                same_origin_meta.get(field)
            ):
                evidence["metadata_matches"][field] = {
                    "value": target_meta[field],
                    "description": description
                }
                
                # Author and last_modified_by are strong indicators
                if field in ["author", "last_modified_by"]:
                    evidence["strong_indicators"].append({
                        "type": "metadata_match",
                        "field": field,
                        "value": target_meta[field],
                        "explanation": f"Identical {description} found in both target and same-origin files"
                    })
        
        # Find shared paths
        evidence["shared_paths"]["user_paths"] = find_matching_paths(
            target_binary["user_paths"],
            same_origin_binary["user_paths"]
        )
        
        evidence["shared_paths"]["template_paths"] = find_matching_paths(
            target_binary["template_paths"],
            same_origin_binary["template_paths"]
        )
        
        evidence["shared_paths"]["system_markers"] = find_matching_paths(
            target_binary["system_markers"],
            same_origin_binary["system_markers"]
        )
        
        # Analyze shared binary signatures
        target_sigs = set(sig["signature"] for sig in target_binary.get("binary_signatures", []))
        same_origin_sigs = set(sig["signature"] for sig in same_origin_binary.get("binary_signatures", []))
        
        shared_sigs = target_sigs & same_origin_sigs
        for sig in shared_sigs:
            evidence["binary_signatures"].append({
                "signature": sig,
                "explanation": "Identical binary signature found in both target and same-origin files"
            })
        
        # Analyze the evidence
        for path_type, paths in evidence["shared_paths"].items():
            for path in paths:
                path_lower = path.lower()
                
                # Definitive markers (user-specific paths with usernames)
                if ('/users/' in path_lower or r'c:\users' in path_lower) and any(
                    part for part in path_lower.split('/') if len(part) > 2
                    and not part.startswith('.')
                    and not any(common in part for common in ['public', 'shared', 'default', 'common'])
                ):
                    evidence["definitive_markers"].append({
                        "type": "user_specific_path",
                        "value": path,
                        "explanation": "Identical user-specific path with username found in both target and same-origin files"
                    })
                
                # Strong indicators (template paths, application paths)
                elif any(marker in path_lower for marker in [
                    'template', '.dotx', '.dotm', 'normal.dot',
                    'application data', 'appdata', 'microsoft/word'
                ]):
                    evidence["strong_indicators"].append({
                        "type": "shared_template",
                        "value": path,
                        "explanation": "Identical template or application-specific path found"
                    })
                
                # Potential indicators (system paths, temp paths)
                else:
                    evidence["potential_indicators"].append({
                        "type": "shared_path",
                        "value": path,
                        "explanation": "Identical system path or marker found"
                    })
        
        # Check for identical styles and fonts (potential indicators)
        shared_styles = set(target_content["styles"]) & set(same_origin_content["styles"])
        if shared_styles:
            evidence["potential_indicators"].append({
                "type": "shared_styles",
                "value": list(shared_styles),
                "explanation": "Identical document styles found in both target and same-origin files"
            })
        
        shared_fonts = set(target_content["fonts_used"]) & set(same_origin_content["fonts_used"])
        if shared_fonts:
            evidence["potential_indicators"].append({
                "type": "shared_fonts",
                "value": list(shared_fonts),
                "explanation": "Identical fonts found in both target and same-origin files"
            })
        
        return evidence

    def compare_files(self) -> Dict[str, Any]:
        """Compare the target file with same-origin file."""
        logging.debug("Starting file comparison...")
        
        # Analyze target file
        logging.debug(f"Analyzing target file: {self.target_path}")
        target_data = {
            "metadata": self.extract_metadata(self.target_path),
            "content": self.analyze_content(self.target_path),
            "ooxml": self.analyze_ooxml_structure(self.target_path),
            "binary": self.analyze_binary_content(self.target_path)
        }
        logging.debug("Target file analysis complete")
        
        # Analyze same-origin file
        logging.debug(f"Analyzing same-origin file: {self.same_origin_path}")
        same_origin_data = {
            "metadata": self.extract_metadata(self.same_origin_path),
            "content": self.analyze_content(self.same_origin_path),
            "ooxml": self.analyze_ooxml_structure(self.same_origin_path),
            "binary": self.analyze_binary_content(self.same_origin_path)
        }
        logging.debug("Same-origin file analysis complete")
        
        # Find origin evidence
        logging.debug("Finding origin evidence...")
        evidence = self.find_origin_evidence()
        
        # Perform statistical analysis
        logging.debug("Starting statistical analysis...")
        try:
            statistical_results = self.statistical_analyzer.analyze_similarity(
                target_data["content"],
                same_origin_data["content"]
            )
            logging.debug("Statistical analysis complete")
        except Exception as e:
            logging.error(f"Error in statistical analysis: {str(e)}")
            statistical_results = {"error": str(e)}
        
        return {
            "target_metadata": target_data["metadata"],
            "same_origin_metadata": same_origin_data["metadata"],
            "origin_evidence": evidence,
            "statistical_analysis": statistical_results
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate the complete analysis report and summary."""
        # Validate files first
        self.validate_files()
        
        # Analyze documents
        target_data = self.analyze_document(self.target_path)
        same_origin_data = self.analyze_document(self.same_origin_path)
        
        # Find origin evidence
        origin_evidence = self.find_origin_evidence()
        
        # Perform statistical analysis
        statistical_analysis = self.statistical_analyzer.analyze_similarity(target_data, same_origin_data)
        
        # Generate report with metadata analysis
        report = {
            "origin_evidence": origin_evidence,
            "statistical_analysis": statistical_analysis,
            "metadata_analysis": {
                "target": target_data.get("metadata", {}),
                "same_origin": same_origin_data.get("metadata", {})
            }
        }
        
        # Generate summary
        summary = self._generate_summary(report)
        report["summary"] = summary
        
        return report

    def _get_cache_key(self, report: Dict[str, Any]) -> str:
        """Generate a cache key for the report."""
        key_data = {
            'target': str(self.target_path),
            'same_origin': str(self.same_origin_path),
            'reference_count': len(self.reference_files),
            'origin_evidence': report['origin_evidence'],
            'statistical': report.get('statistical_analysis', {}),
            'metadata': report.get('metadata_analysis', {})
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _get_cached_summary(self, cache_key: str) -> str:
        """Try to get a cached summary."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            logging.debug("Using cached summary")
            return cache_file.read_text()
        return None

    def _cache_summary(self, cache_key: str, summary: str) -> None:
        """Cache a successful summary."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        cache_file.write_text(summary)
        logging.debug("Cached summary for future use")

    def _generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate a natural language summary using OpenAI API with caching and improved retry logic."""
        cache_key = self._get_cache_key(report)
        cached_summary = self._get_cached_summary(cache_key)
        if cached_summary:
            logging.debug("Using cached summary")
            return cached_summary

        # Create a more concise prompt
        evidence = report["origin_evidence"]
        statistical = report.get("statistical_analysis", {})
        
        prompt = (
            "Analyze Word document forensics evidence:\n"
            f"Definitive markers: {len(evidence.get('definitive_markers', []))}\n"
            f"Strong indicators: {len(evidence.get('strong_indicators', []))}\n"
            f"Potential indicators: {len(evidence.get('potential_indicators', []))}\n\n"
            "Statistical Analysis:\n"
            f"{json.dumps(statistical.get('statistical_summary', {}), indent=2)}\n\n"
            "Provide:\n"
            "1. Conclusion on shared origin\n"
            "2. Evidence strength (definitive/strong/potential)\n"
            "3. Key technical markers\n"
            "4. Statistical confidence\n"
            "5. Caveats\n"
            "6. Further investigation needs"
        )

        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logging.debug(f"Generating AI summary (attempt {attempt + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a digital forensics expert analyzing Word documents for shared origin evidence."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                summary = response.choices[0].message.content
                self._cache_summary(cache_key, summary)
                return summary
                
            except Exception as e:
                logging.error(f"Error generating summary: {str(e)}")
                if attempt == max_retries - 1:
                    logging.warning("Falling back to template summary")
                    return self._generate_template_summary(report)
                time.sleep(2 ** attempt)  # Exponential backoff

        return self._generate_template_summary(report)

    def _generate_template_summary(self, report: Dict[str, Any]) -> str:
        """Fallback template-based summary generator."""
        summary_lines = [
            "Word Document Forensics Analysis Summary",
            "=====================================",
            "",
            "1. Origin Analysis Conclusion:",
        ]
        
        # Extract evidence
        evidence = report["origin_evidence"]
        definitive = evidence.get("definitive_markers", [])
        strong = evidence.get("strong_indicators", [])
        potential = evidence.get("potential_indicators", [])
        
        # Determine overall conclusion
        if definitive:
            summary_lines.append("DEFINITIVE EVIDENCE of shared origin found.")
        elif strong:
            summary_lines.append("STRONG INDICATORS of shared origin detected.")
        elif potential:
            summary_lines.append("POTENTIAL INDICATORS of shared origin present.")
        else:
            summary_lines.append("No clear evidence of shared origin found.")
        
        summary_lines.extend([
            "",
            "2. Evidence Strength:",
            f"- Definitive Markers: {len(definitive)}",
            f"- Strong Indicators: {len(strong)}",
            f"- Potential Indicators: {len(potential)}"
        ])
        
        # Add key markers
        summary_lines.extend([
            "",
            "3. Key Technical Markers:"
        ])
        
        if definitive:
            summary_lines.append("\nDefinitive markers:")
            for marker in definitive:
                summary_lines.append(f"- {marker['explanation']}")
                
        if strong:
            summary_lines.append("\nStrong indicators:")
            for indicator in strong:
                summary_lines.append(f"- {indicator['explanation']}")
                
        if potential:
            summary_lines.append("\nPotential indicators:")
            for indicator in potential:
                summary_lines.append(f"- {indicator['explanation']}")
        
        # Add statistical confidence
        statistical = report.get("statistical_analysis", {})
        if "statistical_summary" in statistical:
            stats = statistical["statistical_summary"]
            confidence = (
                "very high" if stats["likelihood_ratio"] > 0.9 else
                "high" if stats["likelihood_ratio"] > 0.7 else
                "moderate" if stats["likelihood_ratio"] > 0.5 else
                "low"
            )
            summary_lines.extend([
                "",
                "4. Statistical Confidence:",
                f"Based on statistical analysis, there is a {confidence} level of confidence " +
                f"({stats['likelihood_ratio']*100:.1f}%) that the documents share a common origin."
            ])
        else:
            summary_lines.extend([
                "",
                "4. Statistical Confidence:",
                "Statistical analysis was not available for this comparison."
            ])
        
        # Add caveats and recommendations
        summary_lines.extend([
            "",
            "5. Caveats:",
            "- Metadata can be manipulated and should not be solely relied upon",
            "- Statistical analysis is based on available reference documents",
            "- Binary signatures and paths may be system-dependent",
            "",
            "6. Further Investigation Needs:",
            "- Expand reference document dataset for more robust statistical analysis",
            "- Perform deeper analysis of document structure and formatting",
            "- Consider timeline analysis of document modifications",
            "- Analyze any embedded objects or macros"
        ])
        
        return "\n".join(summary_lines)

    def find_docx_files(self, directory: Path) -> List[Path]:
        """Find all .docx files in a directory."""
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        return list(directory.glob("*.docx"))

def main(debug: bool = False) -> None:
    """Main function to run the Word document forensics analysis."""
    try:
        parser = argparse.ArgumentParser(description="Word Document Forensics Analysis")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        args = parser.parse_args()
        
        print("Analyzing documents for shared origin...")
        
        # Set up paths
        project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        input_dir = project_dir / "input"
        target_dir = input_dir / "target"
        same_origin_dir = input_dir / "same_origin"
        reference_dir = input_dir / "reference"
        output_dir = project_dir / "output"
        
        # Create directories if they don't exist
        for dir_path in [input_dir, target_dir, same_origin_dir, reference_dir, output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Find documents
        target_files = list(target_dir.glob("*.docx"))
        same_origin_files = list(same_origin_dir.glob("*.docx"))
        reference_files = list(reference_dir.glob("*.docx"))
        
        if not target_files:
            raise ValueError(f"No target documents found in {target_dir}")
        if not same_origin_files:
            raise ValueError(f"No same-origin documents found in {same_origin_dir}")
            
        target_file = target_files[0]
        same_origin_file = same_origin_files[0]
        
        # Initialize and run analysis
        def run_analysis():
            analyzer = WordForensicAnalyzer(
                target_path=str(target_file),
                same_origin_path=str(same_origin_file),
                reference_files=[str(file) for file in reference_files]
            )
            
            # Generate report
            report = analyzer.generate_report()
            
            # Save report and summary
            report_file = output_dir / "analysis_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
                
            summary_file = output_dir / "analysis_summary.txt"
            with open(summary_file, "w") as f:
                f.write(report["summary"])
                
            return True
            
        # Run with or without debug mode
        if args.debug:
            from debugging.handlers import DebugHandler
            with DebugHandler() as debug_handler:
                success = debug_handler.wrap(run_analysis)()
        else:
            success = run_analysis()
            
        if success:
            print("Analysis complete. Results saved to: output/")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            logging.error(str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

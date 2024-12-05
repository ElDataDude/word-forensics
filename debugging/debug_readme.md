# Word Forensics Debugging Guide

This guide describes the temporary debugging setup used during development of the Word Forensics tool. These debugging components will be removed once the application is working as expected.

## Debug Components

### 1. Debug Handler (`debug_handler.py`)
A custom logging handler that provides detailed logging and AI-assisted analysis of the application's behavior.

Key features:
- Captures detailed logs with timestamps, levels, and context
- Stores logs in JSON format for easy parsing
- Uses GPT-4o-mini to analyze logs and identify issues
- Generates human-readable analysis reports
- Tracks errors, warnings, and successful operations by module

### 2. Debug Analysis Runner (`run_debug_analysis.py`)
A script that runs the entire Word Forensics analysis flow with debug logging enabled.

Features:
- Sets up comprehensive logging
- Runs the complete analysis pipeline
- Handles both single and comparative document analysis
- Generates detailed debug reports

## Using the Debug Setup

1. **Setup**
   ```bash
   # Ensure your .env file contains the OpenAI API key
   OPENAI_API_KEY=your_api_key_here
   ```

2. **Directory Structure**
   ```
   word_forensics/
   ├── debug_logs/        # Debug logs and analysis reports
   ├── reference_docs/    # Reference DOCX files
   ├── output/           # Analysis output
   ├── cache/           # Cache directory
   └── *.docx           # Test documents
   ```

3. **Running Debug Analysis**
   ```bash
   python run_debug_analysis.py
   ```

4. **Debug Output**
   - JSON logs: `debug_logs/debug_log_YYYYMMDD_HHMMSS.json`
   - Analysis report: `debug_logs/debug_analysis_YYYYMMDD_HHMMSS.txt`

## Debug Log Analysis

The debug handler analyzes logs for:
1. Working components
2. Failing components
3. Error patterns
4. Warning patterns
5. Successful operations

## Common Debug Scenarios

1. **No Documents Found**
   - Check if .docx files are present in the working directory
   - Verify file permissions

2. **Analyzer Initialization Failures**
   - Check directory structure
   - Verify reference documents exist
   - Check file permissions

3. **Analysis Failures**
   - Review error logs for specific failure points
   - Check document format and accessibility
   - Verify statistical analysis prerequisites

## Removing Debug Components

Once the application is working as expected:
1. Remove `debug_handler.py` and `run_debug_analysis.py`
2. Remove debug-specific imports from other files
3. Update logging configuration in main application files
4. Remove this DEBUG_README.md

## Note

This debugging setup is temporary and will be removed once the application is stable. Do not rely on these components for production use.

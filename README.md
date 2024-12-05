# Word Document Forensics Tool

A forensic analysis tool specifically designed for analyzing DOCX files (modern Word documents) to determine whether they were created or saved on the same machine. The tool uses statistical analysis and machine learning techniques to identify machine-specific patterns in DOCX file structures, even in the absence of clear metadata like "author" or "company". By comparing against a set of reference DOCX files (known to be from different machines), the tool can demonstrate the statistical unlikelihood of two documents originating from different machines.

## Directory Structure

```
word_forensics/
├── debugging/              # Debug system components
│   ├── handlers/          # Debug handlers and utilities
│   ├── logs/             # Debug logs and analysis reports
│   └── README.md         # Debug system documentation
├── statistical/           # Statistical analysis modules
│   ├── analyzer.py       # Core statistical analysis
│   ├── calculator.py     # Statistical calculations
│   ├── feature_extractor.py  # Feature extraction
│   └── metadata_pairwise_analyzer.py  # Metadata analysis
├── tests/                # Test files and test data
├── input/                # Input documents
│   ├── target/          # Place the primary DOCX file under analysis here
│   ├── same_origin/     # Place DOCX files suspected to share the same origin here
│   └── reference/       # Place known different-origin reference DOCX files here
├── output/              # Analysis results will be saved here
├── cache/               # Cache for analysis results and AI summaries
├── reference_docs/      # Reference documents for analysis
├── .env                # Environment file containing OpenAI API key
└── forensic_analysis.py  # Main entry point
```

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
# .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

2. Set up your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Place your DOCX files in their respective directories:
   - Primary DOCX file → `input/target/`
   - Suspected same-origin DOCX file → `input/same_origin/`
   - Reference DOCX files → `input/reference/`

2. Run the analysis:
```bash
# Normal mode - minimal console output
python forensic_analysis.py

# Debug mode - comprehensive logging and analysis
python forensic_analysis.py --debug
```

The tool will:
- Automatically detect and analyze only DOCX files in each input directory
- Use the first DOCX file found in each directory for analysis
- Generate detailed reports in the `output` directory
- In debug mode, generate additional logs and analysis in `debugging/logs/`

## Analysis Output

The tool performs comprehensive analysis including:
- Metadata extraction and comparison
- Content similarity analysis
- Binary signature detection
- Origin evidence detection with confidence levels
- AI-assisted interpretation of findings

Results are saved as:
- `output/<filename>_analysis.json`: Detailed technical analysis
- `output/<filename>_summary.txt`: Human-readable summary
- When in debug mode:
  - `debugging/logs/debug_<timestamp>.log`: Detailed debug logs
  - `debugging/logs/debug_analysis_<timestamp>.txt`: AI-powered log analysis

## Debug Mode

The tool includes a comprehensive debugging system that can be activated with the `--debug` flag. This system:

1. **Non-Intrusive Logging**
   - Captures detailed information about the analysis process
   - Maintains minimal console output
   - Stores all debug information in dedicated log files

2. **Performance Tracking**
   - Monitors execution time of key operations
   - Tracks resource usage
   - Identifies potential bottlenecks

3. **AI-Powered Log Analysis**
   - Automatically analyzes debug logs
   - Provides insights about errors and warnings
   - Suggests potential improvements
   - Summarizes the most important events

4. **Debug Reports**
   - Generated in `debugging/logs/`
   - Include both raw logs and analyzed summaries
   - Preserve full context for troubleshooting

The debug system is designed to be:
- Non-intrusive to normal operation
- Comprehensive in data collection
- Helpful for development and troubleshooting
- Easy to activate when needed

## How It Works

The Word Forensics tool uses advanced statistical and forensic techniques to analyze DOCX files and determine whether they were created or modified on the same machine. Here's how the analysis works:

### 1. Document Analysis Layers

#### Machine Signature Analysis
- Identifies machine-specific patterns in DOCX file structure and formatting
- Detects system-level markers that persist regardless of content
- Analyzes template and default setting patterns unique to specific machines

#### Metadata Analysis
- Extracts and compares document properties while accounting for potential manipulation
- Identifies consistent patterns in how the machine writes metadata
- Detects system-specific timestamps and encoding patterns

#### Content-Independent Analysis
- Analyzes DOCX file structure independently of user-created content
- Compares binary patterns that are machine-specific
- Examines formatting and style defaults set by the system

#### OOXML Structure Analysis
- Analyzes internal XML structure patterns specific to machine configurations
- Compares relationship patterns between document parts
- Examines content type definitions and machine-specific properties

### 2. Statistical Analysis

The tool employs advanced statistical techniques to prove or disprove same-machine origin:
- Statistical comparison against reference DOCX files from known different machines
- Calculation of likelihood ratios for same-machine vs different-machine scenarios
- Confidence intervals based on reference document variations
- Analysis of machine-specific feature distributions

### 3. Evidence Classification

Findings are classified into three confidence levels:
- **Definitive Markers**: Conclusive evidence of same origin
- **Strong Indicators**: High probability of same origin
- **Potential Indicators**: Possible signs of same origin

### 4. AI-Enhanced Analysis

The tool uses OpenAI's API to:
- Identify subtle machine-specific patterns in DOCX file structure
- Generate human-readable interpretations of technical findings
- Analyze the statistical significance of detected patterns

### 5. Caching and Performance

- Document features are cached for faster subsequent analysis
- AI summaries are stored to avoid redundant API calls
- Binary analysis results are cached per unique file

## Advanced Features

### Reference Document Analysis
- Uses multiple reference DOCX files (known to be from different machines) as a baseline
- Establishes statistical significance of machine-specific patterns
- Quantifies the likelihood of detected similarities occurring between documents from different machines
- Provides confidence levels based on reference document variation

### Feature Importance Ranking
- Ranks evidence by statistical significance
- Provides confidence scores for findings
- Highlights most reliable origin indicators

### Error Handling
- Robust handling of malformed documents
- Graceful degradation of analysis capabilities
- Detailed error reporting and logging

## Note on Rate Limits

The tool uses OpenAI's API for generating summaries. If you encounter rate limits:
- The tool will automatically retry with exponential backoff
- You can press Ctrl+C during retry wait to skip to template-based summary
- Successful summaries are cached for reuse

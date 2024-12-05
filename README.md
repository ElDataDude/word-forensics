# Word Document Forensics Tool

A comprehensive forensic analysis tool for comparing Word documents and determining their origin.

## Directory Structure

```
word_forensics/
├── input/
│   ├── target/        # Place the primary document under analysis here
│   ├── same_origin/   # Place documents suspected to share the same origin here
│   └── reference/     # Place known different-origin reference documents here
├── output/            # Analysis results will be saved here
├── .env              # Environment file containing OpenAI API key
└── forensic_analysis.py
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

1. Place your Word documents in their respective directories:
   - Primary document → `input/target/`
   - Suspected same-origin document → `input/same_origin/`
   - Reference document → `input/reference/`

2. Run the analysis:
```bash
python forensic_analysis.py
```

The tool will:
- Automatically detect .docx files in each input directory
- Use the first file found in each directory for analysis
- Generate detailed reports in the `output` directory:
  - JSON report with comprehensive analysis
  - Text summary with key findings and conclusions
  - Cached AI summaries for faster subsequent runs

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

## How It Works

The Word Forensics tool uses advanced statistical and forensic techniques to analyze Word documents and determine their origin. Here's how the analysis works:

### 1. Document Analysis Layers

#### Metadata Analysis
- Extracts and compares document properties (author, timestamps, revision counts)
- Identifies matching patterns in metadata fields
- Detects inconsistencies or modifications in metadata

#### Content Analysis
- Analyzes document structure (paragraphs, sections, styles)
- Compares text content using statistical similarity measures
- Examines formatting patterns and document organization

#### Binary Analysis
- Examines raw .docx file structure and signatures
- Analyzes embedded resources and media
- Detects system markers and template patterns

#### OOXML Structure Analysis
- Analyzes internal XML structure of .docx files
- Compares relationship patterns between document parts
- Examines content type definitions and custom properties

### 2. Statistical Analysis

The tool employs advanced statistical techniques:
- Principal Component Analysis (PCA) for feature extraction
- Z-score calculations for anomaly detection
- Similarity percentile computation
- Likelihood ratio analysis

### 3. Evidence Classification

Findings are classified into three confidence levels:
- **Definitive Markers**: Conclusive evidence of same origin
- **Strong Indicators**: High probability of same origin
- **Potential Indicators**: Possible signs of same origin

### 4. AI-Enhanced Analysis

The tool uses OpenAI's API to:
- Generate human-readable interpretations of technical findings
- Identify subtle patterns in document structure
- Provide context-aware analysis of similarities

### 5. Caching and Performance

- Document features are cached for faster subsequent analysis
- AI summaries are stored to avoid redundant API calls
- Binary analysis results are cached per unique file

## Advanced Features

### Reference Document Analysis
- Uses multiple reference documents to establish baseline patterns
- Calculates statistical significance of similarities
- Identifies common vs. unique document characteristics

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

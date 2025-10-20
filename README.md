# Balancing Accuracy and Flexibility in Clinical NER: Token Classification vs. Seq2Seq Llama Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper "Balancing Accuracy and Flexibility in Clinical NER: Token Classification vs. Seq2Seq Llama Models," which presents a systematic comparison of two Named Entity Recognition (NER) approaches for Spanish clinical text using Llama models.

## Abstract

This study systematically compares token classification versus sequence-to-sequence approaches in Llama models for Spanish clinical Named Entity Recognition. We evaluated four Llama models (2-7B, 3-8B, 3.1-8B, 3.2-3B) using token classification after continual pre-training on Spanish/English/bilingual corpora, and seq2seq with LoRA adaptation on the PharmaCoNER dataset. 

**Key Findings:**
- Token classification reached 75.44% F1 after bilingual continual pre-training
- Seq2seq achieved 53.30% F1, showing high precision (80.40%) but lower recall (39.90%)
- The 22-point gap quantifies the precision-flexibility trade-off for healthcare IT decision-makers

This work is the first to use decoder models with the PharmaCoNER corpus and establishes evidence-based selection criteria between architectural approaches.

## Repository Structure

```
Data/
  mixData.py                        # Utility for mixing/processing datasets

Generative/
  basePromptInstruction.txt         # Base prompt template for seq2seq approach
  prompts.ipynb                     # Notebook for generating prompts
  codeInstrution.py                 # Instruction tuning implementation
  infer_entities.py                 # Entity extraction inference script
  lengthAnalysis.py                 # Analysis of text/entity length distributions
  runAnalysis.sh                    # Script to run length analysis
  runInference.sh                   # Script to run inference for seq2seq models
  runInstruction.sh                 # Script to run instruction tuning

Seq2Seq/
  Continual/                        # Implementation of continual pre-training
    convert_fabric_to_hf_models.py  # Convert trained models to HF format
    fabric_code_llama.py            # Lightning Fabric training code
    llama_model.json                # Llama model configuration
    preprocessing_causal_lm.py      # Data preprocessing for causal LM
    runContinual.sh                 # Script to run continual pre-training
    runConversion.sh                # Script to run model conversion
    models/                         # Model definition modules
      __init__.py
      models_class.py               # Model class definitions
    utils/                          # Utility functions
      logger.py                     # Logging utilities
      speed_monitor.py              # Training speed monitoring

  Fine-tuning/                      # Token classification fine-tuning
    inference.py                    # Inference for token classification
    llama_pharmacoNER.py            # PharmaCoNER dataset handler
    modeling_llama.py               # Custom Llama modeling code
    runInference.sh                 # Script to run token classification inference
    runTraining.sh                  # Script to run token classification training
```

## Key Contributions

1. **First Use of Decoder Models**: This is the first work to utilize decoder-type models with the PharmaCoNER corpus.
2. **Architectural Trade-off Quantification**: Systematic comparison of token classification vs. seq2seq approaches.
3. **Clinical Deployment Guidance**: Evidence-based decision criteria for selecting NER architectures.
4. **Evaluation Framework**: Comprehensive evaluation methodology for generative medical NER.
5. **Bilingual Domain Adaptation Evidence**: Quantification of computational requirements and training stability.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Lightning 2.0+
- NVIDIA GPUs with at least 24GB VRAM for token classification and 40GB for continual pre-training

### Installation

```bash
git clone https://github.com/edugredu/tokenClassificationVSseq2seq.git
cd tokenClassificationVSseq2seq
pip install -r requirements.txt  # Note: Create this file with your dependencies
```

### Continual Pre-training

To run continual pre-training with Llama models on medical corpora:

```bash
cd Seq2Seq/Continual
./runContinual.sh
```

### Token Classification Fine-tuning

To fine-tune the models for token classification:

```bash
cd Seq2Seq/Fine-tuning
./runTraining.sh
```

### Sequence-to-Sequence Instruction Tuning

To run instruction tuning for the seq2seq approach:

```bash
cd Generative
./runInstruction.sh
```

### Inference

For token classification inference:
```bash
cd Seq2Seq/Fine-tuning
./runInference.sh
```

For sequence-to-sequence inference:
```bash
cd Generative
./runInference.sh
```

## Results

### Token Classification Results

| Language | Epoch | Base model | Precision | Recall | F1 |
|----------|-------|------------|-----------|--------|-----|
| Not continued | - | Llama 2 | 77.22% | 72.46% | 74.77% |
| Spanish + English | 2 | Llama 2 | 77.36% | 73.61% | **75.44%** |
| English | 2 | Llama 2 | 76.02% | 72.08% | 74.00% |
| Spanish | 1 | Llama 2 | 77.30% | 73.17% | 75.18% |

### Sequence-to-Sequence Results

| Base model | Precision | Recall | F1 | JSON Parse | Halluc. Rate | Boundary Acc. | Format Comp. |
|------------|-----------|--------|-----|------------|--------------|---------------|--------------|
| Llama 3.1 | **80.40%** | 39.90% | **53.30%** | 87.20% | 14.45% | 64.56% | 99.54% |
| Llama 3.2 | 66.70% | **40.10%** | 50.10% | **90.00%** | 15.22% | **67.93%** | 99.56% |
| Llama 3 | 67.90% | 39.00% | 49.60% | 86.58% | **11.31%** | 62.44% | **100.00%** |
| Llama 2 | 45.50% | 13.20% | 20.40% | 73.20% | 13.35% | 41.33% | 97.81% |

## Citation

To be added upon publication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PharmaCoNER corpus and evaluation library
- DailyMed by the National Library of Medicine
- CIMA by the Spanish Agency of Medicines and Medical Devices
- University of Alicante and Lancaster University for supporting this research

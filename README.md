# Evaluating Causal Decoder-Only Architectures for Spanish Clinical Named Entity Recognition: Performance Boundaries and Asymmetric Domain Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper "Evaluating Causal Decoder-Only Architectures for Spanish Clinical Named Entity Recognition: Performance Boundaries and Asymmetric Domain Adaptation" which presents a systematic comparison of two Named Entity Recognition (NER) approaches for Spanish clinical text using Llama models.

## Repository Structure

```
Data/
  mixData.py                        # Utility for mixing/processing datasets

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

Generative/
  basePromptInstruction.txt         # Base prompt template for seq2seq approach
  codeInstrution.py                 # Instruction tuning implementation
  infer_entities.py                 # Entity extraction inference script
  lengthAnalysis.py                 # Analysis of text/entity length distributions
  prompts.ipynb                     # Notebook for generating prompts
  runAnalysis.sh                    # Script to run length analysis
  runInference_continuedBase.sh     # Script for inference using the trained models previously continually pre-trained
  runInference_instructedBase.sh    # Script for inference using the trained models taking published instructed-tuned models as base
  runInstruction_continuedBase.sh   # Script for instruction tuning using continually pre-trained models as base
  runInstruction_instructedBase.sh  # Script for instruction tuning using published instructed-tuned models as base
  summarize_inference_metrics.sh    # Script to summarize inference metrics

Seq2Seq/
  Fine-tuning/                      # Token classification fine-tuning
    inference.py                    # Inference for token classification
    llama_pharmacoNER.py            # PharmaCoNER dataset handler
    modeling_llama.py               # Custom Llama modeling code
    runTraining.sh                  # Script to run token classification training and inference
```

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
cd Seq2Seq
./runTraining.sh
```

### Sequence-to-Sequence Instruction Tuning

To run instruction tuning for the seq2seq approach:

```bash
cd Generative
./runInstruction_continuedBase.sh
./runInstruction_instructedBase.sh
```

### Inference

For token classification inference, it is included in the `runTraining.sh` script after training.

For sequence-to-sequence inference:
```bash
cd Generative
./runInference_continuedBase.sh
./runInference_instructedBase.sh
```

## Citation

To be added upon publication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PharmaCoNER corpus and evaluation library
- DailyMed by the National Library of Medicine
- CIMA by the Spanish Agency of Medicines and Medical Devices
# Translational_transformer_from_scratch

This repository implements a Transformer-based neural machine translation model for translating sentences from Italian to English. The project uses PyTorch and related libraries, demonstrating the training process, validation, and sample translation results.

## Features
Custom Transformer architecture (encoder-decoder) implemented from scratch in PyTorch.

Utilizes the opus_books dataset for bilingual sentence pairs.

Tokenization using the Hugging Face tokenizers library.

Training and validation with detailed progress output.

Evaluation metrics: Character Error Rate (CER), Word Error Rate (WER), and BLEU score.

GPU acceleration support.

## Installation
Install required dependencies:

```bash
pip install datasets tokenizers torch torchtext torchmetrics tensorboard
```
Note: The code is tested with torch==2.3.0 and torchtext==0.18.0. If you encounter version conflicts, uninstall previous versions and install the specified ones:

```python
pip uninstall -y torch torchtext
pip install torch==2.3.0 torchtext==0.18.0
```

Run the main training script (see transformer.ipynb for reference):

```
python transformer.py
```
## Configuration:
The script uses a default configuration for batch size, epochs, model dimensions, and dataset. You can modify these in the get_config() function.

## Model Architecture
Tokenization: Word-level tokenizers with special tokens [UNK], [PAD], [SOS], [EOS].

Embedding: Learnable input embeddings and positional encodings.

Encoder/Decoder: Multi-layer (default 6) encoder and decoder stacks with multi-head self-attention and feed-forward blocks.

Training: Cross-entropy loss with label smoothing, Adam optimizer.

Decoding: Greedy decoding for inference.

## Example Results
Below are sample validation results from the model during training. For each example, the SOURCE is the English sentence, the TARGET is the correct Italian translation, and the PREDICTED is the model's output.

![Screenshot 2025-07-01 002030](https://github.com/user-attachments/assets/3e74daae-de90-4a50-9977-af060105b223)


## Training Progress
Training runs for a configurable number of epochs (default: 10).

After each epoch, validation samples are printed, showing SOURCE, TARGET, and PREDICTED translations.

Loss per epoch and evaluation metrics are logged.

## File Structure
transformer.ipynb: Main notebook with full code and explanations.

Model weights and tokenizers are saved automatically during training.

## Customization
Dataset: Change the dataset or language pair by modifying the datasource, lang_src, and lang_tgt fields in the config.

Model Size: Adjust d_model, number of layers, or sequence length in the config for larger or smaller models.

## Notes
The current model outputs are not perfect and can be improved with further training, more data, or hyperparameter tuning.

The provided results show the model's predictions compared to the ground truth for qualitative evaluation.

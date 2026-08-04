# Experiment Protocol

This run uses Llama-3.1-8B-Instruct and the 21 LongBench tasks with the repository's official task-specific prompts, max generation lengths, and scorer.
KIVI and PatternKV use the paper-v2-aligned 2-bit G128/R128 quantization configuration.
The input cap is changed from the paper's approximately 31.5K setting to 8192 tokens because this worker is a single RTX 4090 D 24GB server.
Therefore this experiment must be described as an 8K-capped reproduction, not a strict 31.5K paper reproduction.
Samples shorter than 8192 tokens keep their natural length; longer samples use middle truncation.
The current formal run scope is 21 tasks x 50 samples per task x 3 methods, per the latest user instruction.

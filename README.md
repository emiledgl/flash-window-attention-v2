# Flash Window Attention V2: Accelerating Swin Transformer V2 for Faster Inference and Reduced Memory Footprint

**Flash Window Attention V2** is a triton implementation of the scaled cosine attention occuring in the [Swin transformer V2](https://arxiv.org/abs/2111.09883) attention mechanism. his implementation builds upon the principles introduced in the original [Flash Attention 2 paper](https://arxiv.org/abs/2307.08691), specifically adapting them for the unique requirements of Scaled Cosine (Window) Attention.

## Features

The Swin Transformer V2 is a powerful architecture, but its attention mechanism can be a bottleneck in terms of speed and memory, specially with increasing patch size and window size. By leveraging Triton, Flash Window Attention V2 significantly improves performance by:

* **Reducing memory comsuption:** Following the Flash Attention paradigm, we minimize costly reads and writes to global GPU memory, mainly in the backward pass, where the memory usage is reduced by a factor of 2.
* **Significant Forward and Backward Speedup:** Directly translates to higher throughput for image processing during both training and inference, achieving faster processing for your current models or, alternatively, train and deploy more complex and powerful models while maintaining previous throughput levels.

The reduced memory footprint and the faster inference speed allows for training and deploying larger models or using bigger batch sizes, pushing the boundaries of what's possible on current hardware.

This repository offers a comprehensive and highly optimized implementation of the **scaled cosine attention mechanism** for Swin Transformer V2, implemented entirely using Triton GPU kernels. Key features include:

* **Q and K Normalization:** Ensures stable training and performance.
* **Bias & Masking:** Supports various attention bias configurations and orrectly handles attention masks for windowed attention.
* **Differentiable Logit Scale:** Addresses the unique scaling factor of Swin Transformer V2 by enabling logit scale backward pass.
* **Flexible Dimensions:** Our kernels are designed to be highly versatile, working seamlessly with:
    * **Any Sequence Length (Window Size):** Adaptable to different window configurations.
    * **Any Head Dimension (superior to 16):** Supports a wide range of model architectures.
* **High Numerical Precision:** The implementation maintains very low error compared to the classic PyTorch window attention, and retains the same error level as what you'd expect from `torch.compile` optimizations.
* **Mixed Precision Support:** The kernels efficiently operate in both float16 and bfloat16, enabling faster computation and reduced memory usage on compatible hardware.

## Requirements

- Python
- PyTorch
- Triton

## Performance


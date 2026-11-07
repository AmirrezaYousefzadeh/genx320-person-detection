# TensorFlow Lite Micro with CMSIS-NN kernels

Generated from https://github.com/tensorflow/tflite-micro at commit 1a349f9f7ab91e46b47b987ded4316f186d5e824
(2026-10-29) with:

    python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
      --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m7+fp OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_TOOLCHAIN_ROOT=/usr/bin/" \
      <out_dir>

Unchanged. `firmware/tflm.mk` builds it into `build/libtflm.a`. Licences: Apache 2.0 (TFLM, CMSIS, CMSIS-NN,
flatbuffers), BSD (kissfft); see the LICENSE files.

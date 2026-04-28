# Gemma 2 (2B -it) GGUF Model Files

### Since the finetuned Gemma 2 (2B -it) model GGUF Files have a relatively large size and cannot be directly uploaded in GitHub, I have inserted links in this README to my Google Drive Folder, where the model files are uploaded. They can be downloaded and directly implemented in the inference scripts.

### There are 4 GGUF Format model files suitable for different types of host machines

1. [sentineledge-gemma2-2b-q4_0](https://drive.google.com/file/d/1Ufp9uS3LoUdJPlrQPpEKHkDkWESOhBq-/view?usp=drive_link) - Best for Edge Hardware Inference (ARM Processors, Raspberry Pi) (Lower Accuracy, Faster Inference)
2. [sentineledge-gemma2-2b-q4_k_m](https://drive.google.com/file/d/14Hejc8hS-e6RWoTZT7SwljXkKFDH3Pts/view?usp=drive_link) - Best for High End Edge Hardware Inference (SBCs with CUDA, Nvidia Jeston) (Better Accuracy, Slighly lesser tokens/second)
3. [sentineledge-gemma2-2b-q5_k_m](https://drive.google.com/file/d/1-hqkpsFbgo4uApCuHjoncMP4RWr7YRYl/view?usp=drive_link) - Best for Standard CPU Inference (General Purpose CPU) (Higher Accuracy, Slower Inference)
4. [sentineledge-gemma2-2b-q8_0](https://drive.google.com/file/d/1Q2IpChEJZNKfyzJ5oQmY6p0BZFX3MWLq/view?usp=drive_link) - Best for High End CPU Inference (Servers, Centraized Inference) (Highest Accuracy, Slowest Inference)

### All the Accuracy and Inference behaviour was tested on a standard laptop CPU

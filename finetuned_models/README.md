# Gemma 2 (2B -it) GGUF Model Files

## Since the finetuned Gemma 2 (2b -it) model GGUF Files have a large size and cannot be directly uploaded in GitHub, I have inserted links to my Google Drve Folder, where the model fils are uploaded. They can be downloadeda dn directly implemented in the inference scripts

## There are 4 GGUF Format model files suitable for different types of host machines

1. sentineledge-gemma2-2b-q4_0 - Best for Edge Hardware Inference (ARM Processors, Raspberry Pi) (Lower Accuracy, Faster Inference)
2. sentineledge-gemma2-2b-q4_k_m - Best for High End Edge Hardware Inference (SBCs with CUDA, Nvidia Jeston) (Better Accuracy, Slighly lesser tokens/second)
3. sentineledge-gemma2-2b-q5_k_m - Best for Standard CPU Inference (General Purpose CPU) (Higher Accuracy, Slower Inference)
4. sentineledge-gemma2-2b-q8_0 - Best for High End CPU Inference (Servers, Centraized Inference) (Highest Accuracy, Slowest Inference)
5. 
## All the Accuracy and Inference behaviour was tested on a standard laptop CPU

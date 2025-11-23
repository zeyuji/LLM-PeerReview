export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download sentence-transformers/all-mpnet-base-v2 --local-dir ./LLM_Models/Sentence_Embedding_Models/all-mpnet-base-v2

# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download microsoft/deberta-v3-large --local-dir ./LLM_Models/deberta-v3-large

modelscope download --model="zl2272001/deberta-v3-large" --local_dir ./LLM_Models/deberta-v3-large

# Meta-Llama-3.1-8B-Instruct
modelscope download --model="LLM-Research/Meta-Llama-3.1-8B-Instruct" --local_dir ./LLM_Models/New_7B/Meta-Llama-3.1-8B-Instruct

# Mistral-7B-Instruct-v0.3
modelscope download --model="AI-ModelScope/Mistral-7B-Instruct-v0.3" --local_dir ./LLM_Models/New_7B/Mistral-7B-Instruct-v0.3

# Qwen2-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./LLM_Models/New_7B/Qwen2-7B-Instruct

# Qwen2.5-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./LLM_Models/New_7B/Qwen2.5-7B-Instruct

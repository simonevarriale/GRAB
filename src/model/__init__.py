from .baseline_llm import BaselineLLM
from .soft_prompt_llm import TableSoftPromptLLM
from .grab_models import GrabSingleTable, GrabMultiTable
from src.global_path import model_dir

load_model = {
    'base_llm': BaselineLLM,
    'soft_prompt_llm': TableSoftPromptLLM,
    'grab_single_table': GrabSingleTable,
    'grab_multi_table': GrabMultiTable,
}

llama_model_path = {
    'qwen3_0.6b': f'{model_dir}/Qwen/Qwen3-0.6B',
    'qwen3_4b': f'{model_dir}/Qwen/Qwen3-4B-Base',
    'qwen3_14b': f'{model_dir}/Qwen/Qwen3-14B-Base',
    'qwen3.5_4b': f'{model_dir}/Qwen/Qwen3.5-4B-Base',
    'qwen3.5_27b': f'{model_dir}/Qwen/Qwen3.5-27B',
    'gemma3_1b_pt': f'{model_dir}/google/gemma-3-1b-pt',
    'gemma3_4b_pt': f'{model_dir}/google/gemma-3-4b-pt',
    'llama3.2_3b': f'{model_dir}/meta-llama/Llama-3.2-3B',
}

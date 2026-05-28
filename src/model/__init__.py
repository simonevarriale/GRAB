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
    'qwen3_4b': f'{model_dir}/Qwen/Qwen3-4B-Base',
    'qwen3_14b': f'{model_dir}/Qwen/Qwen3-14B-Base',
}

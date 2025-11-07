"""
Main pipeline for ARC task solving using execution-based similarity.

Pipeline: Program Similarity → Pattern Discovery → Code Generation
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import os
import sys
from src.llm_client import LLMArguments, LLMClient
# print("Script starting...", flush=True)
# sys.stdout.flush()

from src.vlm_prompter import VLMPrompter
from src.vlm_client import VLMConfig, create_client, BaseVLMClient
from src.utils.library import ProgramLibrary, calculate_grid_similarity
from src.utils.dsl import *
from src.utils.constants import *
import argparse
from dotenv import load_dotenv
import pickle
from src.main_utils import process_directory, process_directory_fs
# from utils.render_legacy import grid_to_base64_png_oai_content
parser = argparse.ArgumentParser()
parser.add_argument('--expe_name', type=str, default='test')
parser.add_argument('--data_dir', type=str, default='/home/flowers/work/llms_ftw/tasks/evaluation/')
parser.add_argument('--model_name_or_path', type=str, default='openai/gpt-4-vision-preview')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--top_p', type=float, default=0.8)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--reasoning_effort', type=str, default="low")
parser.add_argument('--path_save', type=str, default='/home/flowers/work/llms_ftw/save_data/test.pkl')
parser.add_argument('--mode', type=str, default='nir')
parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, help="fp8")

args = parser.parse_args()

llm_args = LLMArguments()
llm_args.temperature = args.temperature
llm_args.top_k = args.top_k
llm_args.top_p = args.top_p
llm_args.model_name_or_path = args.model_name_or_path
llm_args.gpu = args.gpu
llm_args.reasoning_effort = args.reasoning_effort  
llm_args.fp8 = args.fp8
llm_client = LLMClient(llm_args)


load_dotenv()
api_key = None
api_base = "http://localhost:8000/v1"
model = "Qwen/Qwen2.5-7B-Instruct"
PROVIDER = "qwen"
vlm_config_phase1 = VLMConfig(
    api_key=api_key,
    model=model,
    api_base=api_base,
    max_tokens=16384,  # Longer for analysis
    save_prompts=False,
    prompt_log_dir="prompts_testing"
)
vlm_config_phase2 = VLMConfig(
    api_key=api_key,
    model=model,
    api_base=api_base,
    max_tokens=8192   # Shorter for code gen
)
vlm_client_phase1 = create_client(PROVIDER, config=vlm_config_phase1)
# print("VLM client created", flush=True)

vlm_client_phase2 = create_client(PROVIDER, config=vlm_config_phase2)
prompter = VLMPrompter(use_vision=False)

library = ProgramLibrary()  # Auto-loads from solvers.py

if args.mode == 'fs':
    out = process_directory_fs(args.data_dir,
                     llm_client,
                     prompter,
                     )
elif args.mode == 'nir':
    out = process_directory(args.data_dir,
                            llm_client,
                            prompter,
                            library)
else:
    raise ValueError(f"Unknown mode: {args.mode}")
with open(args.path_save, 'wb') as f:
    pickle.dump(out, f)
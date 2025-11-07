import copy
from tenacity import retry, wait_exponential, wait_random
from concurrent.futures import ThreadPoolExecutor

import subprocess
import numpy as np
from openai import OpenAI, AzureOpenAI
import requests
from requests.exceptions import RequestException
import time
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.DataTrainingArguments
    """

    model_name_or_path: str = field(
        default="/home/flowers/work/hf/Qwen3-4B-Instruct-2507",#"/home/flowers/work/hf/Qwen2.5-Coder-3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    base_url: Optional[str] = field(
        default="http://localhost:8000",
        metadata={
            "help": "base url for vllm server"
        },
    )
    api_key: Optional[str] = field(
        default="None",
        metadata={
            "help": "api key "
        },
    )
    gpu: Optional[int] = field(
        default = 1,
        metadata={
            "help": "number of gpus to use (vllm)"
        },
    )
    temperature: Optional[float] = field(
        default = 1.0,
        metadata={
            "help": "temperature"
        },
    )
    temperature_labeller: Optional[float] = field(
        default = 0.,
        metadata={
            "help": "temperature labeller (semantic descriptor)"
        },
    )
    min_p: Optional[float] = field(
        default = 0.00,
        metadata={
            "help": "min_p"
        },
    )
    top_p: Optional[float] = field(
        default = 1.,
        metadata={
            "help": "top_p"
        },
    )
    top_k: Optional[int] = field(
        default = -1,
        metadata={
            "help": "top_k"
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "frequency penalty"
        },
    )
    enable_thinking: bool = field(
        default=True,
        metadata={
            "help": "enable thinking for the LLM (for hybrid model, e.g qwen3)"
        },
    )
    reasoning_effort: Optional[str] = field(
        default="high",
        metadata={
            "help": "reasoning effort minimal, low, medium, high"
        },
    )
    max_tokens: Optional[int] = field(
        default = -1,
        metadata={
            "help": "max tokens -1 for no limit"
        },
    )
    max_model_length: Optional[int] = field(
        default = 50000,
        metadata={
            "help": "max context size"
        },
    )
    swap_space: Optional[float] = field(
        default=5,
        metadata={
            "help": "swap space (RAM memory for cache)"
        }
    )
    azure: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use azure if True else use local vllm"
        },
    )
    local_server: Optional[bool] = field(
        default=True,
        metadata={
            "help": "use local_server "
        },
    )
    fp8: bool = field(
        default=True,
        metadata={
            "help": "use fp8 if True"
        },
    )
    gpu_memory: float = field(
        default=0.9,
        metadata={
            "help": "gpu memory to use for vllm server"
        },
    )
    sglang: bool = field(
        default=True,
        metadata={
            "help": "use sglang if True "
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": """log level for sglang {critical,error,warning,info,debug,trace}
            for vllm server {debug,info,warning,error,critical,trace}"""
        },
    )
    ep_moe: bool = field(
        default=False,
        metadata={
            "help": "enable EP for MoE"
        },
    )
    kwargs_engine: str = field(
        default="",
        metadata={
            "help": "additional kwargs for engine, e.g. --kwargs-engine='--enable-reasoning '"
        },
    )
    llm_seed: int = field(
        default=42,
        metadata={
            "help": "random seed"
        },
    )
    debug_skip_init_llm: bool = field(
        default=False,
        metadata={
            "help": "skip initialization of LLM"
        },
    )
class Response:
    def __init__(self, response: list, logprobs):
        self.response = response
        self.logprobs = logprobs

def launch_vllm_serv(model_path: str, gpu: int = 1, max_model_length=20000, port: int = 8000, fp8: bool = False, gpu_memory=0.9, seed: int = 0, log_level="info", add_yarn=False, kwargs_engine=""):
    command = f"vllm serve {model_path} --tensor-parallel-size {gpu} --max-model-len {max_model_length}  --port {port} --gpu-memory-utilization {gpu_memory} --seed {seed} --trust-remote-code --uvicorn-log-level {log_level} "
    if fp8:
        command += "--quantization fp8 "
    list_mistral = ["Mistral-Small-3.2-24B-Instruct-2506","Mistral-Large-Instruct","Codestral-22B-v0.1","Devstral-Small-2505","Magistral-Small-2506"] 
    for model_name in list_mistral:
        if model_name in model_path:
            command += "--tokenizer_mode mistral --config_format mistral --load_format mistral "

    # for qwq and qwen 3 model
    if add_yarn:
        base_model_len = 32768
        if max_model_length < base_model_len:
            pass
        elif max_model_length < 2* base_model_len:
            command += """--rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' --max-model-len 65536 """
        elif max_model_length < 4* base_model_len:
            command +="""--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 """
    command += kwargs_engine
    server_process = execute_shell_command(
        command
    )
    print(command)
    # stuff to add later
    # --uvicorn-log-level {debug,info,warning,error,critical,trace}
    # --reasoning-parser
    # --enable-reasoning
    return server_process

def launch_sglang_serv(model_path: str, gpu: int = 1, max_model_length=20000, port: int = 8000, fp8: bool = False, gpu_memory=0.9, seed: int = 0, log_level="info", add_yarn=False, kwargs_engine=""):
    command = f"python -m sglang.launch_server --model-path {model_path} --tp {gpu} --port {port} --mem-fraction-static {gpu_memory} --random-seed {seed} --host 0.0.0.0 --log-level {log_level} --trust-remote-code "
    if "fp8" in model_path:
        fp8 = False
    if fp8:
        command += "--quantization fp8 "

    # for qwq and qwen 3 model
    if add_yarn:
        base_model_len = 32768
        if max_model_length < base_model_len:
            command += f"--context-length {max_model_length} "
        elif max_model_length < 2* base_model_len:
            command += '--json-model-override-args '+ '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'+' --context-length 65536 '
        elif max_model_length < 4* base_model_len:
            command += '--json-model-override-args '+'{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'+' --context-length 131072 '
    else:
        command += f"--context-length {max_model_length} "
    command += kwargs_engine
    server_process = execute_shell_command(
        command
    )
    print(command)
    # stuff to add later
    # --uvicorn-log-level {debug,info,warning,error,critical,trace}
    # --reasoning-parser
    # --enable-reasoning
    return server_process

def launch_sglang_serv_multi_node(model_path: str, gpu: int = 1, max_model_length=20000, port: int = 8000, port_multinode:int = 5000, fp8: bool = False, gpu_memory=0.9, seed: int = 0, log_level="info", add_yarn=False, ep_moe=False, kwargs_engine=""):
    tp = gpu
    
    worker_num = int(os.environ.get('SLURM_NNODES', 2))
    n_nodes = worker_num
    SLURM_JOB_NODELIST = os.environ.get('SLURM_JOB_NODELIST')

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    nodes_result = subprocess.run(['scontrol', 'show', 'hostnames', SLURM_JOB_NODELIST], 
                                capture_output=True, text=True)

    # Split the output into individual node names
    nodes = [node.strip() for node in nodes_result.stdout.strip().split('\n') if node.strip()]
    head_node = nodes[0]

    # Get the IP address of the head node
    head_node_ip_result = subprocess.run(['srun', '--nodes=1', '--ntasks=1', '-w', head_node, 
                                        'hostname', '--ip-address'], 
                                        capture_output=True, text=True)
    head_node_ip = head_node_ip_result.stdout.strip()

    # Handle potential space-separated IP addresses (IPv4/IPv6) - take the first one
    # head_node_ip = head_node_ip.split()[0]

    # Set environment variable
    os.environ['SGLANG_HOST_IP'] = head_node_ip
    print(f"Head node: {head_node}, Head node IP: {head_node_ip}")

    head_env = os.environ.copy()
    head_env['OUTLINES_CACHE_DIR'] = f"/tmp/{SLURM_JOB_ID}_0"


    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Job {SLURM_JOB_ID} started ...")
    model = model_path
    if "fp8" in model_path.lower():
        fp8 = False

    command_sglang = f"python3 -m sglang.launch_server --model-path {model} --tp {tp} --port {port} --mem-fraction-static {gpu_memory} --random-seed {seed} --host 0.0.0.0 --log-level {log_level} --trust-remote-code "

    if fp8:
        command_sglang += "--quantization fp8 "

    # for qwq and qwen 3 model
    if add_yarn:
        base_model_len = 32768
        if max_model_length < base_model_len:
            command_sglang += f"--context-length {max_model_length} "
        elif max_model_length < 2* base_model_len:
            command_sglang += '--json-model-override-args '+ '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}'+' --context-length 65536 '
        elif max_model_length < 4* base_model_len:
            command_sglang += '--json-model-override-args '+'{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'+' --context-length 131072 '
    else:
        command_sglang += f"--context-length {max_model_length} "

    command_sglang += f"--dist-init-addr {head_node_ip}:{port_multinode} --nnodes {n_nodes} "
    if ep_moe:
        command_sglang += "--enable-ep-moe "
    command_sglang += kwargs_engine
    head_bash_command = f"""echo "BEGIN_IP on Head Node:" && hostname -I && echo "END_IP on Head Node" && \
{command_sglang} --node-rank 0"""

    print(f"Head Node command: {head_bash_command}")
    head_process = subprocess.Popen([
        'srun', '--nodes=1', '--ntasks=1', '-w', head_node, 
        'bash', '-c', head_bash_command
    ], env=head_env)

    HEAD_PID = head_process.pid

    # --- Give Head Node Time to Initialize ---
    print("Waiting for head node to initialize...")
    time.sleep(10)  # Adjust this time if necessary

    # --- Launch Worker Nodes ---
    
    worker_processes = []

    # Loop starts from 1 because 0 is the head node
    for i in range(1, worker_num):
        node_i = nodes[i]
        print(f"STARTING WORKER {i} (Rank {i}) at {node_i}")
        
        worker_env = os.environ.copy()
        worker_env['OUTLINES_CACHE_DIR'] = f"/tmp/{SLURM_JOB_ID}_{i}"

        worker_bash_command = f"{command_sglang} --node-rank {i}"

        print(f"Worker {i} command: {worker_bash_command}")
        worker_process = subprocess.Popen([
            'srun', '--nodes=1', '--ntasks=1', '-w', node_i,
            'bash', '-c', worker_bash_command
        ], env=worker_env)
        
        worker_processes.append(worker_process)

    # Optional: Wait for all processes to complete or handle them as needed
    # For example, to wait for all processes:
    # head_process.wait()
    # for worker_process in worker_processes:
    #     worker_process.wait()

    # Or to keep the main script running while background processes execute:
    print("All processes launched. Main script continuing...")



    return head_process

def check_server_run(model_path, port, server_process,vllm=False):
    """Check if the server is running and serving the correct model.
    Needed when launching multiple inference processes on a cluster.
    """
    try:
        wait_for_server(f"http://localhost:{port}")
        time.sleep(15)
        if vllm:
            req= f"http://localhost:{port}/v1/models"
        else:
            req= f"http://localhost:{port}/get_model_info"
        response = requests.get(
            req,
            headers={"Authorization": "Bearer None"},
        )
        if vllm:
            model_id_serv = response.json()["data"][0]["id"]
        else:
            model_id_serv = response.json()["model_path"]
            good_model = response.json()["model_path"] == model_path
        print("model_id_serv", model_id_serv)
        print("model_path", model_path)
        good_model = model_id_serv == model_path
        print("good_model", good_model)
        if not good_model:
            raise Exception("wrong model")
    except:
        return False
    is_running = server_process.poll() is None
    print("is_running", is_running)
    if not is_running:
        return False
    return True

class LLMClient:
    def __init__(self, llm_args):

        # init cfg generation
        self.llm_args = llm_args
        cfg_generation = {"model": self.llm_args.model_name_or_path, "temperature": self.llm_args.temperature}
        if self.llm_args.max_tokens != -1:
            cfg_generation["max_tokens"] = self.llm_args.max_tokens
        if self.llm_args.min_p != 0:
            if "extra_body" not in cfg_generation:
                cfg_generation["extra_body"] = {}

            cfg_generation["extra_body"]["min_p"] = self.llm_args.min_p
            cfg_generation["min_p"] = self.llm_args.min_p
        if self.llm_args.top_k != -1:
            if "extra_body" not in cfg_generation:
                cfg_generation["extra_body"] = {}
            cfg_generation["extra_body"]["top_k"] = self.llm_args.top_k
        if self.llm_args.top_p != 1:
            cfg_generation["top_p"] = self.llm_args.top_p
        if self.llm_args.presence_penalty != 0:
            cfg_generation["presence_penalty"] = self.llm_args.presence_penalty

        self.model_path = self.llm_args.model_name_or_path
        self.cfg_generation = cfg_generation
        self.base_url = self.llm_args.base_url
        self.api_key = self.llm_args.api_key
        self.timeout = 60*60*4 # 4 h timeout
        self.gpu = self.llm_args.gpu
        self.max_model_length = self.llm_args.max_model_length
        self.azure = self.llm_args.azure
        self.local_server = self.llm_args.local_server
        self.kwargs_engine = self.llm_args.kwargs_engine
        # should add vllm or sglang option
        model_lower = self.model_path.lower()

        # qwen3 for specific stuff link to qwen3 (yarn, hybrid thinking, etc.) 
        self.qwen3 = "qwen3" in model_lower 
        self.think_stop_tag = "</think>"
        if "Magistral-Small-2507" in model_lower:
            self.think_stop_tag = "[/THINK]"

        self.openai_model = "gpt-oss" in model_lower
        if self.openai_model:
            if not "extra_body" in self.cfg_generation:
                self.cfg_generation["extra_body"] = {}
            self.cfg_generation["extra_body"].update({"skip_special_tokens": False})
            self.think_stop_tag = "<|end|><|start|>assistant<|channel|>final<|message|>"
        
        self.seed = self.llm_args.llm_seed
        self.fp8 = self.llm_args.fp8
        if "fp8" in model_lower:
            self.fp8 = False
        self.gpu_memory = self.llm_args.gpu_memory
        self.sglang = self.llm_args.sglang
        self.log_level = self.llm_args.log_level # default log level for sglang
        self.enable_thinking = self.llm_args.enable_thinking 
        self.ep_moe = self.llm_args.ep_moe
        if self.qwen3: 
            if "coder" in model_lower  or "instruct" in model_lower or "thinking" in model_lower:
                self.qwen3 = False

        # if not enable_thinking: # as default, enable_thinking is True
        if not "extra_body" in self.cfg_generation:
            self.cfg_generation["extra_body"] = {}
        
        if "DeepSeek-V3.1".lower() in model_lower:
            self.cfg_generation["extra_body"].update({"chat_template_kwargs":{"thinking": self.enable_thinking}})
        else:
            self.cfg_generation["extra_body"].update({"chat_template_kwargs":{"enable_thinking": self.enable_thinking}})
        
        self.reasoning_parser = ""

        if not self.local_server:
            if "gpt-5" in model_lower:
                self.cfg_generation["reasoning_effort"] = self.llm_args.reasoning_effort
            del self.cfg_generation["extra_body"]
        if self.llm_args.debug_skip_init_llm:
            print("Skipping initialization of LLM as per debug flag.")
        else:
            self.init_client()

    def init_client(self):
        if self.local_server:
            # stuff for slurm protection in case of multiple jobs
            port = np.random.randint(30000,30100)
            # TODO: vllm/ SGLang
            n_nodes = int(os.environ.get('SLURM_NNODES', 1))
            if self.sglang:
                if n_nodes > 1:
                    port_multinode = np.random.randint(5000,7000)
                    server_process = launch_sglang_serv_multi_node(model_path=self.model_path, gpu= self.gpu,
                                                    max_model_length=self.max_model_length, port= port,port_multinode= port_multinode,
                                                    fp8 = self.fp8, gpu_memory=self.gpu_memory, 
                                                    seed = self.seed, log_level = self.log_level,
                                                    add_yarn=self.qwen3 or "qwq" in self.model_path.lower(),
                                                    ep_moe=self.ep_moe, kwargs_engine=self.kwargs_engine)
                else:
                    server_process = launch_sglang_serv(model_path=self.model_path, gpu= self.gpu,
                                                    max_model_length=self.max_model_length, port= port,
                                                    fp8 = self.fp8, gpu_memory=self.gpu_memory, 
                                                    seed = self.seed, log_level = self.log_level,
                                                    add_yarn=self.qwen3 or "qwq" in self.model_path.lower(),
                                                    kwargs_engine=self.kwargs_engine)
                
            else:
                server_process = launch_vllm_serv(model_path=self.model_path, gpu= self.gpu,
                                                max_model_length=self.max_model_length, port= port,
                                                fp8 = self.fp8, gpu_memory=self.gpu_memory, 
                                                seed = self.seed, log_level = self.log_level,
                                                add_yarn = self.qwen3 or "qwq" in self.model_path.lower(),
                                                kwargs_engine=self.kwargs_engine)
            print("check server run 0")
            if n_nodes > 1:
                n_tries = 4
            else:
                n_tries = 1
            for i_try in range(n_tries):
                is_running = check_server_run(self.model_path,port,server_process,vllm=not self.sglang)
                if is_running:
                    break

            

            if not is_running:
                for i_try in range(1,3):
                    port += 1
                    if self.sglang:
                        server_process = launch_sglang_serv(model_path=self.model_path, gpu= self.gpu,
                                                        max_model_length=self.max_model_length, port= port,
                                                        fp8 = self.fp8, gpu_memory=self.gpu_memory, seed = self.seed)
                    else:
                        # launch vllm server
                        server_process = launch_vllm_serv(model_path=self.model_path, gpu= self.gpu,
                                        max_model_length=self.max_model_length, port= port,
                                        fp8 = self.fp8, gpu_memory=self.gpu_memory, seed = self.seed)
                    print("check server run ", i_try)
                    is_good = check_server_run(self.model_path,port,server_process,vllm=not self.sglang)
                    if is_good:
                        break
            else:
                print(' /!\ Server is running /!\ ')
            self.base_url=f"http://localhost:{port}/v1" 
            self.api_key="None"
            is_good = check_server_run(self.model_path,port,server_process,vllm=not self.sglang)
            self.server_process = server_process
            if not is_good:
                raise Exception("wrong model")

        
        
        api_key = None
        if self.api_key == "":
            self.api_key = None
        if self.azure: 
            self.client = AzureOpenAI(base_url=self.base_url, api_key=self.api_key,timeout=self.timeout,api_version="2025-01-01-preview",)
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key,timeout=self.timeout)

        print("Server is up and running")


    def get_reasing_parser(self):
        """not use for now, (just split based on "</think>')"""
        model = self.model_path.lower()
        self.reasoning_parser =""
        if self.qwen3 and self.enable_thinking:
            self.reasoning_parser = "qwen3" 
        if "qwq" in model:
            self.reasoning_parser = "deepseek-r1"
        if "r1" in model:
            self.reasoning_parser = "deepseek-r1"
        if "deepcoder" in model:
            self.reasoning_parser = "deepseek-r1"

    def extract_reasoning_response(self, response: str) -> tuple[str,str]:
        """Extract reasoning from the response"""
        reasoning = None
        sol = response
        think_stop_tag = self.think_stop_tag
        if think_stop_tag in response:
            reasoning = response.split(think_stop_tag)[0].strip() + think_stop_tag
            sol = response.split(think_stop_tag)[1].strip()
        return reasoning, sol

    def extract_reasoning_response_batch(self, responses: list[str]) -> list[tuple[str,str]]:
        """Extract reasoning from the response"""
        list_reasoning = []
        list_responses = []
        for full_response in responses:
            reasoning, sol = self.extract_reasoning_response(full_response)
            list_reasoning.append(reasoning)
            list_responses.append(sol)
        return list_reasoning, list_responses

    def formating_chat_prompt(self, list_prompt_str: list[str]) -> list[list[dict]]:
        """Format list of prompt string to chat prompt"""
        list_prompt_chat=[]
        for prompt in list_prompt_str:
            # check whether I used syst prompt or not
            list_prompt_chat.append([{"role": "user", "content": prompt}])
        return list_prompt_chat

    def terminate(self):
        try:
            terminate_process(self.server_process) 
        except Exception as e:
            print(f"Error terminating server process: {e}")

    def multiple_completion(self, batch_prompt,n=1,temperature=None):
        if isinstance(batch_prompt, list) and isinstance(batch_prompt[0], str):
            batch_prompt = self.formating_chat_prompt(batch_prompt)
        batch_prompt = self.add_reasoning_system_prompt(batch_prompt)
        return get_multiple_completions(self.client, batch_prompt, cfg_generation=self.cfg_generation,n=n,temperature=temperature)
    
    def generate(self, prompts,n=1,temperature=None):
        return self.multiple_completion(prompts,n=n,temperature=temperature)

    def add_reasoning_system_prompt(self, batch_prompt):
        """Prepend the system prompt to each message in the batch."""
        model = self.model_path.lower()
        if "magistral" in model:
            if "2506" in model:
                magistral_sys_prompt = magistral_2506_sys_prompt
            elif "2507" in model:
                magistral_sys_prompt = load_magistral_2507_system_prompt()
            else:
                raise ValueError(f"Unknown magistral model: {model}")
            sys_prompt = magistral_sys_prompt
        elif "llama-3_3-nemotron-super" in model:
            sys_prompt =  f"detailed thinking {self.enable_thinking}"
            if "v1_5" in model:
                if self.enable_thinking:
                    sys_prompt =  f""
                else:
                    sys_prompt =  f"/no_think"
        elif "gpt-oss" in model:
            sys_prompt = f"Reasoning: {self.llm_args.reasoning_effort}"
        else:
            return batch_prompt
        patched_batch = []
        for message in batch_prompt:
            # Find the user message content
            if "Magistral-Small-2507".lower() in model:
                patched_message = sys_prompt + message
            else:
                patched_message = [{"role": "system", "content": sys_prompt}] + message
            patched_batch.append(patched_message)
        return patched_batch

    def get_embeddings(self,batch_prompt) ->list[list[float]]:
        cfg_generation = {"model":self.llm_args.model_name_or_path}
        list_out = get_multiple_embeddings(self.client,batch_prompt,cfg_generation)
        return list_out

def is_server_up(base_url):
    attempt=50
    while attempt>0:
        try:
            # Try to connect to the health endpoint or root endpoint
            if "/v1" in base_url:
                base_u=base_url.replace("/v1","")
            else:
                base_u=base_url
            response = requests.get(f"{base_u}/health", timeout=5)

            print(response.status_code)
            flag = response.status_code == 200
            if flag:
                print("="*20)
                print("serv succesfuly initializes")
                print("="*20)
                return True
            else:
                raise
        except RequestException as e:
            print(e)
            flag= False
        attempt-=1
        time.sleep(20)


@retry(wait=wait_exponential(multiplier=1, min=10, max=600)+wait_random(min=0, max=1))
def get_embedding(client, cfg_generation: dict, input: list) -> list[float]:
    kwargs = cfg_generation.copy()
    try:
        response = client.embeddings.create(
            input=input,
            **kwargs
        )
    except Exception as e:
        print("completion problem: ", e)
        too_long = "longer than the model's context length" in e.body["message"]
        if too_long:
            print(too_long)
            # return [e.body["message"]] * n
        return [None] 

    out = response.data[0].embedding
    return out


def get_multiple_embeddings(client, batch_prompt: list[list], cfg_generation: dict={}, max_workers=90)->list[list[float]]:
    """Get multiple completions from OpenAI API"""
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    
    
    cfg_gen_copy= {"model":cfg_generation["model"]}
    completions = []
    count=0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_batch in chunks(batch_prompt, max_workers):
            
            for message in sub_batch:
                count+=1
                kwargs = {
                    "client": client,
                    "input": message,
                    "cfg_generation": cfg_gen_copy,
                }
                future = executor.submit(get_embedding, **kwargs)
                completions.append(future)

            print(f"send {count} / {len(batch_prompt)} messages")

    # Retrieve the results from the futures
    out_n = [future.result() for future in completions]
    return out_n


# @retry(wait=wait_exponential(multiplier=1, min=30, max=600)+wait_random(min=0, max=1))
@retry(wait=wait_exponential(multiplier=1, min=10, max=600)+wait_random(min=0, max=1))
def get_completion(client, cfg_generation: dict, messages: list, temperature=None, n=1) -> list[str]:
    """Get completion(s) from OpenAI API"""
    kwargs = cfg_generation.copy()
    if temperature is not None:
        kwargs["temperature"] = temperature
    # if "min_p" in kwargs:
    #     del kwargs["min_p"]
        # closed API doesn't support min_p 
    kwargs["n"] = n

    try:
        completion = client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    except Exception as e:
        print("completion problem: ", e)
        too_long = "longer than the model's context length" in e.body["message"]
        if too_long:
            print(too_long)
            # return [e.body["message"]] * n
        return [None] * n

        raise e
    

    out = [choice.message.content for choice in completion.choices]
    return out


    # list_response = []
    # #TODO: check that
    # for completion_out in completion.choices:
    #     list_response.append(completion_out.message.content)
    # # response = completion.choices[-1].message.content
    
    # return Response(list_response,logprobs)
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_multiple_completions(client, batch_prompt: list[list], cfg_generation: dict={}, max_workers=90, temperature=None, n=1)->list[list[str]]:
    """Get multiple completions from OpenAI API"""
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    
    completions = []
    count=0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_batch in chunks(batch_prompt, max_workers):
            
            for message in sub_batch:
                count+=1
                kwargs = {
                    "client": client,
                    "messages": message,
                    "cfg_generation": cfg_generation,
                    "temperature": temperature,
                    "n": n
                }
                future = executor.submit(get_completion, **kwargs)
                completions.append(future)

            print(f"send {count} / {len(batch_prompt)} messages")

    # Retrieve the results from the futures
    out_n = [future.result() for future in completions]
    return out_n

# def get_multiple_completions_judge(guided_choice,*kwargs)->list[list[str]]:
#     copy.deepcopy()


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)

def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)

                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)

# stuff to terùminate the process and release the port
import threading
import signal
import psutil
import os
import sys
import weakref
process_socket_map = weakref.WeakKeyDictionary()


def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """

    kill_process_tree(process.pid)

    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)

def release_port(lock_socket):
    """
    Release the reserved port by closing the lock socket.
    """
    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")



def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass

magistral_2506_sys_prompt = """A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown and Latex to format your response. Write both your thoughts and summary in the same language as the task posed by the user.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.

Problem:"""

magistral_2507_sys_prompt = """First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response."""

def load_magistral_2507_system_prompt() -> dict[str]:
    # file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # with open(file_path, "r") as file:
    #     system_prompt = file.read()
    system_prompt = magistral_2507_sys_prompt
    index_begin_think = system_prompt.find("[THINK]")
    index_end_think = system_prompt.find("[/THINK]")

    return [{
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:index_begin_think]},
            {
                "type": "thinking",
                "thinking": system_prompt[
                    index_begin_think + len("[THINK]") : index_end_think
                ],
                "closed": True,
            },
            {
                "type": "text",
                "text": system_prompt[index_end_think + len("[/THINK]") :],
            },
        ],
    }]
if __name__ == "__main__":
    # test multi nodes
    model = "/lustre/fsn1/projects/rech/imi/uqv82bm/hf/GLM-4.5-Air-FP8"#Qwen2.5-14B-Instruct" #DeepSeek-R1-0528"
    cfg_generation = {"model": model, "temperature": 0.6}
    gpu=4
    local_server = True
    fp8=False
    sglang = True
    ep_moe = True
    llm = LLMClient(model=model, cfg_generation=cfg_generation, gpu=gpu, local_server=local_server, fp8=fp8, sglang=sglang, ep_moe=ep_moe)
    test_messages = [
        {"role": "system", "content": "You are a good assistant"},
        {"role": "user", "content": "Is it chocolatine or pain au chocolat?"},
    ]
    
    out = llm.multiple_completion([test_messages], n=1)
    print(out[0].response)
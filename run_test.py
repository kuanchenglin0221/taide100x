import torch
from transformers import AutoTokenizer, AutoModelForCausalLM ,AutoConfig

import argparse
import time
import os
import logging

from models import NaiveWrapper, HuggingFaceWrapper, FlashInferWrapper
from models.llm.modeling_llama_fi import LlamaForCausalLM


from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from awq.models._config import AwqConfig
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_IPEX,
    WQLinear_Marlin,
    WQLinear_Exllama,
    WQLinear_ExllamaV2,
    WQLinear_GEMVFast,
    marlin_post_init,
    exllama_post_init,
    exllamav2_post_init,
    ipex_post_init,
)
from awq.utils.module import (
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
    try_import,
)
from tqdm import tqdm
import gc

from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
def _load_quantized_modules(
        model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
    ):
        # Real quantization of weights
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2 or use_ipex)
        ), "Exllama kernels only support GEMM version."

        # Get blocks of model
        layers = model.model.layers

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            # self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                if version == "marlin":
                    q_linear_module = WQLinear_Marlin
                # elif version == "gemm":
                #     q_linear_module = WQLinear_GEMM
                # elif version == "gemv":
                #     q_linear_module = WQLinear_GEMV
                # elif version == "gemv_fast":
                #     q_linear_module = WQLinear_GEMVFast
                
                q_linear = q_linear_module.from_linear(
                    module, quant_config.w_bit, quant_config.q_group_size, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            if not use_ipex:
                torch.cuda.empty_cache()
            gc.collect()

def load_model(
    llm_path: str,
    mode: str,
    dtype: torch.dtype = torch.float16,
    device: str = "auto",
    q_llm_path: str = None,
    ):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    
    # load LLM
    if mode == "flashinfer":
        llm = LlamaForCausalLM.from_pretrained(
            args.llm_path, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
    )
    else :
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
        )
    
    # load quantized model
    if q_llm_path is not None:
         
        quant_config = AwqConfig.from_pretrained(q_llm_path)

        _load_quantized_modules(llm, quant_config, quant_config.version, False, False, use_ipex=False)
        llm = marlin_post_init(llm)

        model_weights_path = q_llm_path +"/model.safetensors"
        layer_type = "LlamaDecoderLayer"
        device_map = "auto"
        max_memory = None
        offload_folder = None
        torch_dtype = torch.float16
        load_checkpoint_and_dispatch(
                    llm,
                    checkpoint=model_weights_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    no_split_module_classes=[layer_type],
                    offload_folder=offload_folder,
                    dtype=torch_dtype,
        )

    if mode == "naive":
        model = NaiveWrapper()
    elif mode == "huggingface" or mode == "hf":
        model = HuggingFaceWrapper()
    elif mode in ["flashinfer"]:
        model = FlashInferWrapper()
    else:
        raise ValueError("Invalid mode.")
    
    # set model
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)
    model.eval()
    
    return model, tokenizer


def main(args):

     # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)
    
    # deterministic
    torch.manual_seed(0)

    # load model
    print("Loading model...")
    model, tokenizer = load_model(args.llm_path, args.mode, q_llm_path = args.q_llm_path)

    # warm up
    if not args.no_warm_up:
        print("Warming up model...")

        # input message
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = "Hello."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        _  = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)

    # generate response
    print("Generating response...")

    # input message
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    input_message = "What's the best way to start learning a new language?"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_message},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
    prompt = tokenizer.decode(input_ids[0])
    
    start_time = time.time()
    output_ids = model.generate(input_ids, temperature=args.temp, max_new_tokens=args.max_new_tokens, max_length=args.max_length, do_sample=args.do_sample)
    end_time = time.time()
    
    output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])

    if not args.no_print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if not args.no_print_time:
        print("Time:", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--llm-path",
        "-llm",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="LLM model path.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to do sampling. (Default is False)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        help="The mode of model generation.",
    )
    parser.add_argument(
        "-nw",
        "--no-warm-up",
        action="store_true",
        help="Warm up the model.",
    )
    parser.add_argument(
        "-nm",
        "--no-print-message",
        action="store_true",
        help="Print the message.",
    )
    parser.add_argument(
        "-nt",
        "--no-print-time",
        action="store_true",
        help="Record the time.",
    )
    parser.add_argument(
        "--q_llm-path",
        "-q_llm",
        type=str,
        default=None,
        help="quantized LLM model path.",
    )
    args = parser.parse_args()
    
    main(args)
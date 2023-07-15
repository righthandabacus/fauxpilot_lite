import functools
import logging
import random
import string
import time

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache()
def get_codegen():
    """Return the codegen tokenizer and model"""
    logger = logging.getLogger("pilot.model")
    modelname = 'Salesforce/codegen-2B-mono'
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForCausalLM.from_pretrained(modelname,
                                                 #load_in_8bit=True,  # <- needs bitsandbytes and accelerate, and libcudart
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 #device_map="auto"
                                                 ).to(device)
    logger.info("Loaded %s, footprint %s", modelname, model.get_memory_footprint())
    return tokenizer, model


@functools.lru_cache()
def get_codegen2():
    """Return the codegen2 tokenizer and model"""
    logger = logging.getLogger("pilot.model")
    modelname = 'Salesforce/codegen2-1B'
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, revision="main").to(device)
    logger.info("Loaded %s, footprint %s", modelname, model.get_memory_footprint())
    return tokenizer, model


@functools.lru_cache()
def get_codet5():
    """Return the codet5-base tokenizer and model"""
    logger = logging.getLogger("pilot.model")
    modelname = 'Salesforce/codet5-base'
    tokenizer = RobertaTokenizer.from_pretrained(modelname)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(modelname).to(device)
    logger.info("Loaded %s, footprint %s", modelname, model.get_memory_footprint())
    return tokenizer, model


@functools.lru_cache()
def get_codet5p():
    """Return the codet5+ tokenizer and model"""
    logger = logging.getLogger("pilot.model")
    modelname = 'Salesforce/codet5p-770m'
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname,
                                                  #torch_dtype=torch.float16,
                                                  #low_cpu_mem_usage=True,
                                                  trust_remote_code=True).to(device)
    logger.info("Loaded %s, footprint %s", modelname, model.get_memory_footprint())
    return tokenizer, model


def dummy_generate(data):
    """Run dummy model: for fast testing"""

    text = data['prompt']
    completion = "<this is a dummy output>"
    response = {
        "id": 'cmpl-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29)),
        "model": data["model"],
        "object": "text_completion",
        "created": int(time.time()),
        "choices": [{
            'text': completion,
            'index': 0,
            'finish_reason': 'stop',
            'logprobs': None
        }],
        "usage": {
            "completion_tokens": len(completion),
            "prompt_tokens": len(text),
            "total_tokens": len(text)+len(completion),
        }
    }
    return response


def generate(data):
    """Run model"""
    logger = logging.getLogger("pilot.run")
    logger.debug(data)
    # Load model and tokenizer
    if data['model'] == 'codegen':
        tokenizer, model = get_codegen()
    elif data['model'] == 'codegen2':
        tokenizer, model = get_codegen2()
    elif data['model'] == 'codet5':
        tokenizer, model = get_codet5()
    elif data['model'] == 'codet5p':
        tokenizer, model = get_codet5p()
    elif data['model'] == 'dummy':
        return dummy_generate(data)
    else:
        raise NotImplementedError(f"Model {data['model']} unrecognized")

    # get the prompt and prepare parameters
    text = data['prompt']
    logger.debug(text)
    encoding = tokenizer(text, return_tensors="pt").to(device)
    logger.debug(encoding)
    attention_mask = encoding.get("attention_mask")  # TODO ensure not None
    kwargs = {}
    if data['model'] == 'codet5p':
        kwargs.update(encoding)
        kwargs['decoder_input_ids'] = kwargs['input_ids'].clone()
    else:
        kwargs['inputs'] = encoding.input_ids
    if data['suffix']:
        # TODO handle data['suffix'] - need client plugin to support
        raise NotImplementedError(f"suffix is not supported yet")
    # update other input parameters
    if data['temperature'] is not None:
        kwargs['temperature'] = data['temperature']
        if kwargs['temperature'] == 0.0:
            kwargs['temperature'] = 1.0
            kwargs['top_k'] = 1
    else:
        kwargs['temperature'] = 0.2
    if data['max_tokens'] is not None:
        kwargs['max_new_tokens'] = data['max_tokens']
    kwargs['top_p'] = data['top_p'] or 1.0
    if data['presence_penalty'] is not None:
        kwargs['diversity_penalty'] = data['presence_penalty']
    if data['frequency_penalty'] is not None:
        kwargs['repetition_penalty'] = data['frequency_penalty']
    else:
        kwargs['repetition_penalty'] = 1.0
    kwargs['do_sample'] = True
    kwargs['num_return_sequences'] = 1  # what if more than 1?
    kwargs['attention_mask'] = attention_mask
    kwargs['pad_token_id'] = tokenizer.eos_token_id
    # run model and decode generated content
    logger.debug(kwargs)
    generated = model.generate(**kwargs)
    logger.debug(generated)
    prompt_tokens = len(encoding["input_ids"][0])
    generated = generated[0]
    if data['model'] in ['codegen', 'codegen2', 'codet5p']:
        # these models will repeat the prompt
        generated = generated[prompt_tokens:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    completion_tokens = len(generated)
    # prepare response
    response = {
        "id": 'cmpl-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29)),
        "model": data["model"],
        "object": "text_completion",
        "created": int(time.time()),
        "choices": [{
            'text': completion,
            'index': 0,
            'finish_reason': 'length' if completion_tokens==data['max_tokens'] else 'stop',
            'logprobs':None
        }],
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens+completion_tokens,
        }
    }
    return response

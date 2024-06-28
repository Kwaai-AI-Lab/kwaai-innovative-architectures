from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import perf_counter
import pandas as pd
import numpy as np
hf_models = ["facebook/opt-125m","EleutherAI/pythia-160m","openai-community/gpt2",
             "facebook/opt-350m","openai-community/gpt2-medium","EleutherAI/pythia-410m",
             "openai-community/gpt2-large","EleutherAI/pythia-1b",
             "EleutherAI/pythia-1.4b","facebook/opt-1.3b","openai-community/gpt2-xl","EleutherAI/gpt-neo-1.3b",
             "EleutherAI/gpt-neo-2.7b","EleutherAI/pythia-2.8b","facebook/opt-2.7b"]
mamba_models = ["state-spaces/mamba-130m", "state-spaces/mamba2-130m",
                "state-spaces/mamba-370m","state-spaces/mamba2-370m",
                "state-spaces/mamba-790m","state-spaces/mamba2-780m",
                "state-spaces/mamba-1.4b","state-spaces/mamba2-1.3b",
                "state-spaces/mamba-2.8b","state-spaces/mamba2-2.7b",
                ]
with open("speed_test_text.txt","r") as f:
    text = f.read().split("\n")
    text = " ".join(text)

input_token_length = [20, 50, 100, 500, 1000]
output_token_length = [20, 50, 100, 200, 300, 400, 500, 600, 700]
p_temperature = 0.5
p_top_p = 1.0

def run_hf_inference(pretrained_path, input_length, output_length):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    # Return NaN if the input token and output token exceeds the model's maximum input sequence length
    if input_length + output_length > tokenizer.model_max_length:
        return np.NaN
    model = AutoModelForCausalLM.from_pretrained(pretrained_path)
    encoded_input = tokenizer.encode(text, max_length=input_length, truncation=True, return_tensors="pt")
    tstart = perf_counter()
    output = model.generate(input_ids=encoded_input, max_new_tokens=output_length, temperature=p_temperature, do_sample=True, top_p=p_top_p)
    tstop = perf_counter()
    return tstop - tstart

def run_mamba_inference(pretrained_path, input_length, output_length):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # Return NaN if the input token and output token exceeds the model's maximum input sequence
    if input_length + output_length > tokenizer.model_max_length:
        return np.NaN
    model = MambaLMHeadModel.from_pretrained(pretrained_path, device = 'cuda')
    encoded_input = tokenizer.encode(text, max_length=input_length, truncation=True, return_tensors="pt").to('cuda')
    max_tokens = input_length + output_length
    tstart = perf_counter()
    output = model.generate(encoded_input, max_length= max_tokens, temperature = p_temperature, top_p = p_top_p)
    tstop = perf_counter()
    return tstop - tstart

def main():
    data_rows = []
    for model_path in hf_models:
        for in_token in input_token_length:
            for out_token in output_token_length:
                inference_time = run_hf_inference(model_path, in_token, out_token)
                row = {}
                row["Model Name"] = model_path
                row["Input Tokens"] = in_token 
                row["Output Tokens"] = out_token
                row["Time (second)"] = inference_time
                row["Tokens/Sec"] = out_token / inference_time
                print("Added {0}".format(row))
                data_rows.append(row)
    for model_path in mamba_models:
        for in_token in input_token_length:
            for out_token in output_token_length:
                inference_time = run_mamba_inference(model_path, in_token, out_token)
                row = {}
                row["Model Name"] = model_path
                row["Input Tokens"] = in_token 
                row["Output Tokens"] = out_token
                row["Time (second)"] = inference_time
                row["Tokens/Sec"] = out_token / inference_time
                print("Added {0}".format(row))
                data_rows.append(row)
    df = pd.DataFrame(data_rows)
    df.to_csv("llm_speed_benchark_results.csv", index=False)

if __name__ == '__main__':
    main()
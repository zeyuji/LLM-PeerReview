import argparse
import random
from typing import Dict, List, Union
import numpy as np
import os

from tqdm import tqdm
from openai import OpenAI

from Utils.util import (
    load_data_config,
    load_embedding_input,
    load_response,
    load_reference,
    clean_generations,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_config",
    type=str,
    help="Path to config file. This should be a yaml file.",
)
parser.add_argument(
    "--data_name",
    type=str,
    help="Name of the dataset.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory with data files.",
)
parser.add_argument(
    "--response_dir",
    type=str,
    help="Directory with response files.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    help="Directory to save results to.",
)


check_sys_msg = """
You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. 
This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.
"""

def replace_user_prompt(instruction, output_1, output_2, switch):
    template = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{
    "instruction": "{instruction}",
}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{
    {
        "model_identifier": "model1",
        "output": "{output_1}"
    },
    {
        "model_identifier": "model2",
        "output": "{output_2}"
    }
}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): model1 or model2.

## Best Model Identifier
"""
    template = template.replace("{instruction}", instruction)
    if switch:
        template = template.replace("{output_1}", output_2)
        template = template.replace("{output_2}", output_1)
    else:
        template = template.replace("{output_1}", output_1)
        template = template.replace("{output_2}", output_2)
    return template

def extract_model_identifier(output, switch):
    import re
    
    change_back = {
        "model1": "model2",
        "model2": "model1"
    }

    # Convert to lowercase and slice the last 50 characters of the output
    output_segment = output[-50:].lower()

    # This regex will match 'model1' or 'model2'
    pattern = r'\b(model1|model2)\b'
    matches = re.findall(pattern, output_segment)

    # Check which model appears last in the segment (which means first from the back to front)
    if matches:
        last_match = matches[-1]  # Get the last match found, which is the first from the end
        if switch:
            last_match = change_back[last_match]
        
        return True, last_match
    else:
        # No valid identifier was found
        return False, None


api_key='xxx'
client = OpenAI(api_key=api_key)

def ask_gpt(
    messages: List[Dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    output_text = response.choices[0].message.content
    print(f'GPT_output: {output_text} \n\n')
    
    return output_text

def scorer(response):
    if response == "model1":
        return True
    else:
        return False

def evaluate_gpt_cmp(
    generations: List,
    references: List,
    instructions: List,
) -> float:
    switches = []
    for i in range(len(instructions)):
        switch = random.choice([True, False])
        switches.append(switch)
    # print("switches:", switches)
    
    all_user_prompt = []
    all_messages = []
    for i in range(len(generations)):
        user_prompt = replace_user_prompt(instructions[i], generations[i], references[i], switches[i])
        messages = [
            {"role": "system", "content": check_sys_msg},
            {"role": "user", "content": user_prompt}
        ]
        all_user_prompt.append(user_prompt)
        all_messages.append(messages)

    # print("all_user_prompt:", all_user_prompt)

    scores = []
    for i, messages in enumerate(tqdm(all_messages, desc="Evaluating samples via GPT", unit="sample")):
        flag = False
        for j in range(3):
            response = ask_gpt(messages)

            # print(f"\nsample_{i+1}_response_{j+1}:{response}\n")

            success, result = extract_model_identifier(response, switches[i])

            if success:
                flag = (True if scorer(result) else False)
                break
        scores.append(1 if flag else 0)
    return scores

def evaluate(
    data_config: Dict,
    responses: List,
    references: Union[List[str], List[List[str]]],
    instructions: List,
) -> float:
    task_generations = clean_generations(responses)

    metrics = data_config["metrics"]
    metrics = metrics[0]
    scores = []
    if metrics == "gpt_cmp":
        scores.extend(evaluate_gpt_cmp(
            generations=task_generations, 
            references=references, 
            instructions=instructions))
    else:
        raise ValueError("Unknown metrics")

    assert len(scores) == len(responses)
    return np.mean(scores)

def main(args):
    data_config = load_data_config(args.dataset_config)
    instructions = load_embedding_input(args.data_dir)
    references = load_reference(args.data_dir)
    scores = []

    seed_n = 1
    responses = load_response(args.response_dir, seed_n)
    scores.append(evaluate(
        data_config, responses, references, instructions))

    print("Scores: {}".format(scores))
    print("Mean score: {}".format(np.mean(scores)))

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    with open(args.results_dir + "/" + args.data_name + ".txt", "w") as f:
        f.write("Scores: {}\n".format(scores))
        f.write("Mean score: {}\n".format(np.mean(scores)))
    
if __name__ == "__main__":
    print("Evaluate by GPT...")
    args = parser.parse_args()
    print("You are using args:\n{}".format(args))
    main(args)
    print("#" * 100)
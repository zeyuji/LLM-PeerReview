"""
This script implements evaluation metrics for different tasks, focusing on accuracy calculation for
TriviaQA and GSM8K/MATH datasets. The main goal is to evaluate the correctness of generated responses
against a set of reference answers.
"""

import re
from typing import List


def trivia_qa_acc(generations: List[str], references: List[List[str]]) -> List[int]:
    """
    Compute accuracy for TriviaQA dataset. A generation is considered correct if it contains any of 
    the reference answers. The comparison is case-insensitive.

    Args:
        generations (List[str]): A list of generated answer strings.
        references (List[List[str]]): A list of lists, where each sublist contains reference answers 
                                      for a given query.

    Returns:
        List[int]: A list of 0s and 1s, where 1 indicates a correct generation and 0 indicates an 
                   incorrect generation.
    """
    correct = []
    for gen, refs in zip(generations, references):
        gen_lower = gen.lower()
        # Check if any of the reference answers is present in the generated answer
        if any(ref.lower() in gen_lower for ref in refs):
            correct.append(1)
        else:
            correct.append(0)
    return correct


def gsm8k_acc(generations: List[str], references: List[str]) -> List[int]:
    """
    Compute accuracy for the GSM8K/MATH dataset. The generated response is correct if it either contains
    the reference answer explicitly or matches the last extracted number (if any) from the generation.

    Args:
        generations (List[str]): A list of generated answer strings.
        references (List[str]): A list of reference answer strings.

    Returns:
        List[int]: A list of 0s and 1s, where 1 indicates a correct generation and 0 indicates an 
                   incorrect generation.
    """
    
    def extract_last_number(input_str: str) -> str:
        """
        Extracts the last number from the input string using a regular expression.
        
        Args:
            input_str (str): The input string to search for numbers.

        Returns:
            str or None: The last number found as a string, or None if no number is found.
        """
        pattern = r"\d+\.?\d*"  # Regular expression to match integers and floating-point numbers
        matches = re.findall(pattern, input_str)
        return matches[-1] if matches else None

    correct = []
    no_extract = 0
    for gen, ref in zip(generations, references):
        gen_sentences = gen.split(". ")
        true_flag = False
        answer_flag = False
        
        # Check if the generation contains an explicit answer
        for sentence in gen_sentences:
            if "answer" in sentence.lower():
                answer_flag = True
                gen_lower = sentence.lower().replace(",", "")
                ref_lower = ref.lower().replace(",", "")
                if ref_lower in gen_lower:
                    correct.append(1)
                    true_flag = True
                    break
        
        # If an answer was found explicitly, mark it as correct
        if true_flag:
            continue
        elif answer_flag:
            # If the generation contains an answer but it's incorrect, mark it as incorrect
            correct.append(0)
        else:
            # If no explicit answer is found, attempt to extract numbers and compare
            no_extract += 1
            gen_answer = extract_last_number(gen)
            ref_answer = extract_last_number(ref)
            if gen_answer and ref_answer and gen_answer == ref_answer:
                correct.append(1)
            else:
                correct.append(0)
    
    return correct

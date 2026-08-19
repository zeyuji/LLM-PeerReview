"""
This script implements evaluation metrics for different tasks, focusing on accuracy calculation for
TriviaQA and GSM8K/MATH datasets. The main goal is to evaluate the correctness of generated responses
against a set of reference answers.
"""

import re
import random
from typing import Any, Dict, List, Optional, Sequence

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


FLOAT_COMPARISON_EPSILON = 1e-12


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


def parse_answer(generation: str) -> Optional[str]:
    """
    Extracts the answer from a generation text.
    
    Priority order:
    1. Look for "the answer is X" format
    2. Look for LaTeX expressions
    3. Look for sentences containing "answer"
    4. Fallback: extract last number
    
    Args:
        generation: The generated text to extract answer from
        
    Returns:
        Extracted answer as string, or None if no answer found
    """
    if generation is None:
        return None

    text = generation.strip()
    if not text:
        return None

    # Remove common chat/special tokens that often pollute extraction
    text = re.sub(r"<\|[^>]*\|>", " ", text)  # e.g. <|im_end|>
    text = text.replace("</s>", " ").replace("<s>", " ")
    text = re.sub(r"\s+", " ", text).strip()

    def extract_boxed(s: str) -> Optional[str]:
        # Prefer the last boxed expression (models often restate)
        matches = re.findall(r"\\boxed\{([^}]*)\}", s)
        if matches:
            return matches[-1].strip()
        return None

    def extract_last_number_or_fraction(s: str) -> Optional[str]:
        # Supports ints/floats/scientific and simple fractions like -1/3
        frac_pat = r"-?\d+\s*/\s*-?\d+"
        num_pat = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
        matches = re.findall(rf"(?:{frac_pat})|(?:{num_pat})", s)
        if matches:
            return matches[-1].replace(" ", "")
        return None

    # Priority 1: boxed content anywhere
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    # Priority 2: "the answer is ..." but only keep a short tail and re-extract
    m = re.search(r"\bthe answer is\b\s*(.+)$", text, flags=re.IGNORECASE)
    if m:
        tail = m.group(1).strip()
        # Cut at common hard boundaries to avoid swallowing the whole remainder
        tail = re.split(r"(?:\n|\.|!|\?|</s>|<\|)", tail, maxsplit=1)[0].strip()
        boxed_tail = extract_boxed(tail)
        if boxed_tail is not None:
            return boxed_tail
        last_tail = extract_last_number_or_fraction(tail)
        if last_tail is not None:
            return last_tail
        # If tail is short, keep it as-is (e.g., "3\\text{ cm}")
        if 0 < len(tail) <= 64:
            return tail

    # Priority 3: emulate gsm8k_acc idea — look for a sentence mentioning "answer"
    # Split on sentence-ish boundaries including newlines.
    sentences = re.split(r"[.!?\n]\s*", text)
    for sent in sentences:
        if "answer" not in sent.lower():
            continue
        boxed_sent = extract_boxed(sent)
        if boxed_sent is not None:
            return boxed_sent
        last_sent = extract_last_number_or_fraction(sent)
        if last_sent is not None:
            return last_sent

    # Priority 4: fallback — last number/fraction in full text
    last = extract_last_number_or_fraction(text)
    if last is not None:
        return last

    return None


def normalize_number(answer: str) -> Optional[str]:
    """
    Normalizes a number or mathematical expression string.
    
    Handles:
    - Removing commas (1,000 -> 1000)
    - Standardizing whitespace
    - Removing dollar signs and LaTeX delimiters
    - Converting fractions to decimals when possible
    
    Args:
        answer: The answer string to normalize
        
    Returns:
        Normalized answer string, or None if input is None/empty
    """
    if answer is None or not answer.strip():
        return None
    
    normalized = answer.strip()

    # Remove LaTeX delimiters (remove both sides if present)
    normalized = re.sub(r"^\\\(", "", normalized)
    normalized = re.sub(r"\\\)$", "", normalized)
    normalized = re.sub(r"^\\\[", "", normalized)
    normalized = re.sub(r"\\\]$", "", normalized)
    normalized = re.sub(r"^\$", "", normalized)
    normalized = re.sub(r"\$$", "", normalized)
    
    # Remove commas in numbers
    normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
    
    # Standardize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


def is_mathematically_equivalent(answer1: str, answer2: str, tol: float = 1e-6) -> bool:
    """
    Determines if two answers are mathematically equivalent.
    
    Priority order of comparison methods:
    1. Exact string match (after normalization)
    2. Numeric comparison (for pure numbers)
    3. Symbolic math comparison using sympy (if available)
    
    Args:
        answer1: First answer string
        answer2: Second answer string
        tol: Tolerance for floating point comparisons
        
    Returns:
        True if answers are mathematically equivalent, False otherwise
    """
    if answer1 is None or answer2 is None:
        return False
    
    # Normalize both answers
    norm1 = normalize_number(answer1)
    norm2 = normalize_number(answer2)
    
    if norm1 is None or norm2 is None:
        return False
    
    # Priority 1: Exact string match
    if norm1 == norm2:
        return True
    
    # Priority 2: Numeric comparison
    try:
        # Try to parse both as floats
        num1 = float(norm1)
        num2 = float(norm2)
        # Check if they are approximately equal
        if abs(num1 - num2) <= tol:
            # Check sign consistency
            if (num1 >= 0 and num2 >= 0) or (num1 <= 0 and num2 <= 0):
                return True
    except (ValueError, TypeError):
        pass
    
    # Priority 3: Symbolic math comparison using sympy
    if SYMPY_AVAILABLE:
        try:
            # Try to parse both expressions
            expr1 = None
            expr2 = None
            
            # Try parsing as LaTeX first
            try:
                expr1 = parse_latex(norm1)
                expr2 = parse_latex(norm2)
            except Exception:
                pass
            
            # If LaTeX parsing failed, try sympify
            if expr1 is None or expr2 is None:
                try:
                    expr1 = sympy.sympify(norm1)
                    expr2 = sympy.sympify(norm2)
                except Exception:
                    pass
            
            if expr1 is not None and expr2 is not None:
                # Simplify both expressions
                simplified1 = sympy.simplify(expr1)
                simplified2 = sympy.simplify(expr2)
                
                # Check for exact equality
                if simplified1 == simplified2:
                    return True
                
                # For numeric expressions, try numerical evaluation
                try:
                    val1 = float(simplified1.evalf())
                    val2 = float(simplified2.evalf())
                    if abs(val1 - val2) <= tol:
                        return True
                except Exception:
                    pass
        except Exception:
            pass
    
    return False


def weighted_voting(
    answers_parsed: Sequence[Optional[str]],
    weights: Sequence[float],
    *,
    equivalence_tol: float = 1e-6,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Perform weighted voting over parsed answers.

    Cluster parsed answers by mathematical equivalence, sum each cluster's
    weights, and select the cluster with the highest total weight. Ties are
    broken randomly. None values, which represent unparseable answers, are
    ignored.

    answers_parsed should contain values returned by parse_answer and may
    include None.
    """
    if len(answers_parsed) != len(weights):
        raise ValueError(
            f"answers_parsed length ({len(answers_parsed)}) != weights length ({len(weights)})"
        )

    if rng is None:
        rng = random.Random()

    # Use the first parsed answer in each cluster as its representative.
    representatives: List[str] = []
    cluster_weights: List[float] = []

    for parsed_answer, weight in zip(answers_parsed, weights):
        if parsed_answer is None:
            continue

        assigned = False
        for i, rep in enumerate(representatives):
            if is_mathematically_equivalent(parsed_answer, rep, tol=equivalence_tol):
                cluster_weights[i] += float(weight)
                assigned = True
                break

        if not assigned:
            representatives.append(parsed_answer)
            cluster_weights.append(float(weight))

    if not representatives:
        return {"answer_clusters": {}, "selected_answer": None}

    max_weight = max(cluster_weights)
    candidates = [
        rep
        for rep, w in zip(representatives, cluster_weights)
        if abs(w - max_weight) < FLOAT_COMPARISON_EPSILON
    ]

    selected_answer = rng.choice(candidates) if candidates else None
    answer_clusters = {str(rep): w for rep, w in zip(representatives, cluster_weights)}
    return {"answer_clusters": answer_clusters, "selected_answer": selected_answer}



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

from typing import List
import numpy as np
from jiwer import wer, cer, ExpandCommonEnglishContractions

def normalize_text(text: str) -> str:
    """
    Normalize text
    Args:
        text
    Returns:
        Normalized text
    """
    expander = ExpandCommonEnglishContractions()
    expanded_text = expander(text.lower())
    # print(f"expanded_text: {expanded_text}")
    # expanded_text = text
    return expanded_text

def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Word Error Rate
    Args:
        predictions
        references
    Returns:
        WER
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references have different length")

    total_wer = 0.0

    for pred, ref in zip(predictions, references):
        expanded_ref = normalize_text(ref)
        expanded_pred = normalize_text(pred)
        
        if not expanded_ref:
            if not expanded_pred:
                continue
            total_wer += 1.0
            continue
            
        total_wer += wer(expanded_ref, expanded_pred)
        
    return total_wer / len(predictions)

def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Character Error Rate
    Args:
        predictions
        references
    Returns:
        CER
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references have different length")
        
    total_cer = 0.0

    for pred, ref in zip(predictions, references):
        expanded_ref = normalize_text(ref)
        expanded_pred = normalize_text(pred)
        
        if not expanded_ref:
            if not expanded_pred:
                continue
            total_cer += 1.0
            continue
            
        total_cer += cer(expanded_ref, expanded_pred)
        
    return total_cer / len(predictions)

def calculate_metrics(predictions: List[str], 
                     references: List[str]) -> dict:
    """
    Calculate all evaluation metrics
    Args:
        predictions
        references
    Returns:
        A dictionary containing WER and CER
    """
    metrics = {
        'wer': calculate_wer(predictions, references),
        'cer': calculate_cer(predictions, references)
    }
    return metrics 

if __name__ == "__main__":
    preds = ["LET'S GO"]
    refs = ["LET US GO"]
    
    computed_wer = calculate_wer(preds, refs)
    print("Computed WER:", computed_wer) 
import os
import re
import pandas as pd
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datasets import load_dataset

load_dotenv()


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    with open(prompt_path, "r") as f:
        return f.read()


def get_speech(text: str) -> str:
    """Extract speech from the last <speech></speech> tags."""
    matches = re.findall(r"<speech>\s*(.*?)\s*</speech>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def build_generation_prompt(speech: str, prompt_template: str) -> str:
    """Build the generation prompt from template."""
    return f"{prompt_template}\n<speech>{speech}</speech>"


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content from a specific XML tag."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def parse_maxims(output: str) -> dict:
    """Parse maxim values from output."""
    return {
        "pred_maxim_quality": extract_tag_content(output, "quality"),
        "pred_maxim_quantity": extract_tag_content(output, "quantity"),
        "pred_maxim_relevance": extract_tag_content(output, "relevance"),
        "pred_maxim_manner": extract_tag_content(output, "manner"),
    }


def parse_speech_act(output: str) -> dict:
    """Parse speech act from output."""
    return {
        "pred_speech_act": extract_tag_content(output, "speech_act"),
    }


def get_groundtruth_maxims(sample: dict) -> dict:
    """Get groundtruth maxim values from sample."""
    return {
        "gt_maxim_quality": sample.get("maxim_quality", ""),
        "gt_maxim_quantity": sample.get("maxim_quantity", ""),
        "gt_maxim_relevance": sample.get("maxim_relevance", ""),
        "gt_maxim_manner": sample.get("maxim_manner", ""),
    }


def get_groundtruth_speech_act(sample: dict) -> dict:
    """Get groundtruth speech act from sample."""
    return {
        "gt_speech_act": sample.get("speech_act", ""),
    }


def compare_values(pred: str, gt: str) -> int:
    """Compare prediction and groundtruth (case-insensitive)."""
    return 1 if pred.strip().lower() == gt.strip().lower() else 0


def calculate_f1_multilabel(pred: str, gt: str) -> dict:
    """Calculate precision, recall, F1 for multi-label (comma-separated) values."""
    pred_set = set(p.strip().lower() for p in pred.split(",") if p.strip())
    gt_set = set(g.strip().lower() for g in gt.split(",") if g.strip())

    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1}

    if not pred_set or not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compare_maxims(parsed: dict, groundtruth: dict) -> dict:
    """Compare maxim predictions with groundtruth."""
    return {
        "compare_maxim_quality": compare_values(parsed["pred_maxim_quality"], groundtruth["gt_maxim_quality"]),
        "compare_maxim_quantity": compare_values(parsed["pred_maxim_quantity"], groundtruth["gt_maxim_quantity"]),
        "compare_maxim_relevance": compare_values(parsed["pred_maxim_relevance"], groundtruth["gt_maxim_relevance"]),
        "compare_maxim_manner": compare_values(parsed["pred_maxim_manner"], groundtruth["gt_maxim_manner"]),
    }


def compare_speech_act(parsed: dict, groundtruth: dict) -> dict:
    """Compare speech act prediction with groundtruth (multi-label F1)."""
    metrics = calculate_f1_multilabel(parsed["pred_speech_act"], groundtruth["gt_speech_act"])
    return {
        "speech_act_precision": metrics["precision"],
        "speech_act_recall": metrics["recall"],
        "speech_act_f1": metrics["f1"],
    }


def generate_response(
        llm: ChatOpenAI,
        speech: str,
        prompt_template: str,
        verbose: bool = False,
) -> str:
    """Generate response for a single record."""
    prompt = build_generation_prompt(speech, prompt_template)

    result = llm.invoke([("human", prompt)])
    response = result.content

    if verbose:
        print("\n" + "=" * 80)
        print("GENERATED OUTPUT:")
        print("=" * 80)
        print(response)
        print("=" * 80 + "\n")

    return response


def main(
        output_path: str,
        prompt_path: str,
        output_type: str = "maxim",  # "maxim" or "speech_act"
        test_num: Optional[int] = 10,
        test_size: float = 0.1,
        model_name: str = "gpt-4o",
        verbose: bool = True,
):
    """
    Main function to generate outputs using a prompt template.

    Args:
        output_path: Path to save the output CSV
        prompt_path: Path to the prompt template file
        output_type: Type of output to extract ("maxim" or "speech_act")
        test_num: Number of records to process from top (None = all)
        test_size: Test split ratio for dataset
        model_name: OpenAI model name to use
        verbose: Whether to print generated outputs
    """
    # Load dataset
    dataset = load_dataset("hungphongtrn/proactive-ai-2000")["train"]
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    test_dataset = split_dataset["test"]

    print(f"Loaded {len(test_dataset)} test records")

    # Load prompt template
    prompt_template = load_prompt_template(prompt_path)
    print(f"Loaded prompt from {prompt_path}")

    # Select records from top
    if test_num is not None and test_num < len(test_dataset):
        test_dataset = test_dataset.select(range(test_num))
        print(f"Selected first {test_num} records")
    else:
        print(f"Using all {len(test_dataset)} records")

    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_retries=2,
    )

    # Select parser, groundtruth extractor, and comparator based on output type
    if output_type == "maxim":
        parser = parse_maxims
        gt_extractor = get_groundtruth_maxims
        comparator = compare_maxims
    elif output_type == "speech_act":
        parser = parse_speech_act
        gt_extractor = get_groundtruth_speech_act
        comparator = compare_speech_act
    else:
        raise ValueError(f"Unknown output_type: {output_type}. Use 'maxim' or 'speech_act'.")

    results = []

    for idx, sample in enumerate(tqdm(test_dataset, desc="Generating")):
        speech = sample["user1"]

        if verbose:
            print(f"\n{'#' * 80}")
            print(f"RECORD {idx}")
            print(f"{'#' * 80}")
            print(f"Speech: {speech[:200]}..." if len(speech) > 200 else f"Speech: {speech}")

        # Generate output
        generation = generate_response(
            llm=llm,
            speech=speech,
            prompt_template=prompt_template,
            verbose=verbose,
        )

        # Parse prediction
        parsed = parser(generation)

        # Get groundtruth
        groundtruth = gt_extractor(sample)

        # Compare prediction with groundtruth
        comparison = comparator(parsed, groundtruth)

        # Build result record
        result = {
            "original_index": idx,
            "speech": f"<speech>{speech}</speech>",
            "output": generation,
            **parsed,
            **groundtruth,
            **comparison,
        }
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print accuracy
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if output_type == "maxim":
        compare_columns = [col for col in results_df.columns if col.startswith("compare_")]
        for col in compare_columns:
            accuracy = results_df[col].mean()
            print(f"{col}: {accuracy:.4f} ({results_df[col].sum()}/{len(results_df)})")
        if compare_columns:
            overall_accuracy = results_df[compare_columns].values.mean()
            print(f"\nOverall accuracy: {overall_accuracy:.4f}")

    elif output_type == "speech_act":
        print(f"speech_act_precision: {results_df['speech_act_precision'].mean():.4f}")
        print(f"speech_act_recall: {results_df['speech_act_recall'].mean():.4f}")
        print(f"speech_act_f1: {results_df['speech_act_f1'].mean():.4f}")


if __name__ == "__main__":
    # Configuration
    # output_path = "outputs/generation_maxim.csv"
    # prompt_path = "prompts/prompt_maxim.md"
    # output_type = "maxim"

    output_path = "outputs/generation_speech_act.csv"
    prompt_path = "prompts/prompt_speech_act.md"
    output_type = "speech_act"

    test_num = 100  # Set to None for all records

    main(
        output_path=output_path,
        prompt_path=prompt_path,
        output_type=output_type,
        test_num=test_num,
        model_name="gpt-5",
        verbose=True,
    )
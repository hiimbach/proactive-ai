import os
import re
import json
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_speech(text: str) -> str:
    """Extract speech from the last <speech></speech> tags."""
    matches = re.findall(r"<speech>\s*(.*?)\s*</speech>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()  # Get the last match
    return text.strip()


def get_response(text: str) -> str:
    """Extract response from <response></response> tags."""
    match = re.search(r"<response>\s*(.*?)\s*</response>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_evaluation_prompt(
    speech: str,
    response: str,
) -> str:
    """Build the evaluation prompt for GPT."""
    rubric = """
## Evaluation Rubric

### Dimension 1 – Intent Understanding
Question: Does the model correctly understand what the user wants?
| Score | Label | Description |
|-------|-------|-------------|
| 1 | Poor | Misreads the goal (e.g., treats a complaint as a casual comment or info question). Responds off-topic or answers a different question. |
| 2 | Adequate | Captures the rough intent (e.g., "needs help with internet"), but either (a) misses finer distinctions (Complaint vs Action_Request vs Confusion), or (b) only answers part of the goal. |
| 3 | Strong | Correctly identifies the dominant intent (Complaint, Action_Request, Confirmation, etc.). Tailors the reply directly to that intent (e.g., complaint → apologise + propose fix, not just explanation). |

### Dimension 2 – Emotion Understanding & Alignment
Question: Does the model correctly read and respond to the user's emotional state?
| Score | Label | Description |
|-------|-------|-------------|
| 1 | Poor | Ignores emotion completely or misreads it (e.g., cheerful tone to an angry user). Responds in a way that could escalate frustration or feel cold. |
| 2 | Adequate | Detects broad polarity (positive / negative / neutral). Shows some empathy, but tone may be generic, underplayed, or slightly mismatched. |
| 3 | Strong | Correctly distinguishes types of negative emotion (e.g., frustration vs confusion vs anxiety). Mirrors and de-escalates emotion naturally: frustration → acknowledge + apologise; anxiety → reassure + give clear steps; confusion → normalise + clarify. |

### Dimension 3 – Pragmatic Inference (Implicature & Maxims)
Focus: How well it uses pragmatic cues to refine intent + emotion.
| Score | Label | Description |
|-------|-------|-------------|
| 1 | Poor | Takes everything literally; misses sarcasm, indirect refusals, hints. Treats hedged / vague utterances as if they were clear and complete. |
| 2 | Adequate | Notices some cues (e.g., "sounds annoyed"), but doesn't fully integrate them. Only partially recovers hidden intent (e.g., recognises "unhappy" but not "wants refund"). |
| 3 | Strong | Correctly uses flouted maxims (Quantity, Quality, Relevance, Manner) to sharpen intent + emotion. E.g., sarcasm (Quality flout) → Complaint + Anger; vague "I'm not sure this is working…" (Quantity flout) → Confusion + Help_Request. Gives a reply that clearly addresses the implied meaning, not just the literal words. |

### Dimension 4 – Proactive Fit: Action & Options
Question: Given its reading of intent + emotion, does the model choose good next actions?
| Score | Label | Description |
|-------|-------|-------------|
| 1 | Poor | Either does nothing proactive (only answers literally), or jumps to a strong action without checking what the user wants. Suggested actions don't match the user's real goal or emotional state. |
| 2 | Adequate | Offers some next step, but it may be generic ("Let me know if you need anything else") or only one option with weak confirmation. |
| 3 | Strong | Proactive behaviour is anchored in intent + emotion. E.g., Complaint + Anger → "I'm sorry; I can (1) fix X now or (2) escalate Y — which do you prefer?"; Confusion + Anxiety → "I can (1) walk you through step-by-step or (2) send a short summary." Presents 2–3 clear options and asks the user to choose before executing. |

### Dimension 5 – Conversational Naturalness (Emotion-aware)
Question: Does the reply sound like a natural, emotionally appropriate response?
| Score | Label | Description |
|-------|-------|-------------|
| 1 | Poor | Robotic, overly formal, or clearly templated. Lengthy, cluttered, or hard to follow; ignores the user's emotional tone. |
| 2 | Adequate | Grammatically fine and mostly clear. Style is acceptable but slightly stiff or mismatched to the user's mood. |
| 3 | Strong | Short, clear, and conversational; respects Quantity (enough info, not a wall of text). Tone matches emotion: calm with anxious users, respectful with angry/complaining users. Uses simple, human phrases ("Sounds really frustrating", "Totally get why you'd feel that way"). |
"""

    prompt = f"""You are an expert evaluator for conversational AI responses.

{rubric}

## Task
Evaluate the following AI response based on the rubric above. 

### Input Speech:
{speech}

### Model's Response:
{response}

## Instructions
1. First, provide your thinking process analyzing each dimension. Thinking carefully about each dimension before giving the score. The judgement should be a bit strict, 3.0 is only for a perfect response in that dimension.
2. Then output the scores in the following exact format:

<scores>
intent_understanding: [1-3]
emotion_understanding: [1-3]
pragmatic_inference: [1-3]
proactive_fit: [1-3]
conversational_naturalness: [1-3]
</scores>

Provide your evaluation:
"""
    return prompt


def parse_scores(text: str) -> Dict[str, int]:
    """Parse scores from GPT response."""
    scores = {
        "intent_understanding": 0,
        "emotion_understanding": 0,
        "pragmatic_inference": 0,
        "proactive_fit": 0,
        "conversational_naturalness": 0,
    }

    scores_match = re.search(r"<scores>(.*?)</scores>", text, re.DOTALL | re.IGNORECASE)
    if scores_match:
        scores_text = scores_match.group(1)
        for key in scores.keys():
            pattern = rf"{key}:\s*(\d)"
            match = re.search(pattern, scores_text, re.IGNORECASE)
            if match:
                scores[key] = int(match.group(1))

    return scores


def evaluate_record(
    llm: ChatOpenAI,
    speech: str,
    response: str,
    verbose: bool = False,
) -> Dict:
    """Evaluate a single record using GPT."""
    prompt = build_evaluation_prompt(
        speech=speech,
        response=response,
    )

    result = llm.invoke([("human", prompt)])
    gpt_response = result.content

    if verbose:
        print("\n" + "=" * 80)
        print("GPT EVALUATION OUTPUT:")
        print("=" * 80)
        print(gpt_response)
        print("=" * 80 + "\n")

    scores = parse_scores(gpt_response)
    avg_score = sum(scores.values()) / len(scores) if any(scores.values()) else 0

    return {
        "scores": scores,
        "average_score": avg_score,
        "gpt_raw_output": gpt_response,
    }


def main(
    evaluation_path: str,
    output_path: str,
    test_num: Optional[int] = 10,
    model_name: str = "gpt-4o",
    verbose: bool = True,
    save_detailed_json: bool = True,
):
    """
    Main function to run evaluation.

    Args:
        evaluation_path: Path to the input CSV file
        output_path: Path to save the output CSV
        test_num: Number of records to test from top (None = all)
        model_name: OpenAI model name to use
        verbose: Whether to print GPT raw output during evaluation
        save_detailed_json: Whether to save detailed results to JSON
    """
    # Load data
    df = pd.read_csv(evaluation_path)
    print(f"Loaded {len(df)} records from {evaluation_path}")

    # Select records from top
    if test_num is not None and test_num < len(df):
        df_sample = df.head(test_num).copy()
        print(f"Selected first {test_num} records")
    else:
        df_sample = df.copy()
        print(f"Using all {len(df)} records")

    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_retries=2,
    )

    results = []
    detailed_results = []

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Evaluating"):
        # Get speech
        speech_raw = row.get("speech", "")
        speech = get_speech(speech_raw) if "<speech>" in speech_raw else speech_raw

        # Get response
        generation = row.get("generation", "")
        response = get_response(generation)

        if verbose:
            print(f"\n{'#' * 80}")
            print(f"RECORD {idx}")
            print(f"{'#' * 80}")
            print(f"Speech: {speech[:200]}..." if len(speech) > 200 else f"Speech: {speech}")
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

        # Evaluate
        eval_result = evaluate_record(
            llm=llm,
            speech=speech,
            response=response,
            verbose=verbose,
        )

        # Build result record for CSV
        result = {
            "original_index": idx,
            "speech": speech,
            "response": response,
            "intent_understanding": eval_result["scores"]["intent_understanding"],
            "emotion_understanding": eval_result["scores"]["emotion_understanding"],
            "pragmatic_inference": eval_result["scores"]["pragmatic_inference"],
            "proactive_fit": eval_result["scores"]["proactive_fit"],
            "conversational_naturalness": eval_result["scores"]["conversational_naturalness"],
            "average_score": eval_result["average_score"],
            "gpt_raw_output": eval_result["gpt_raw_output"],
        }
        results.append(result)

        # Build detailed record for JSON
        detailed_record = {
            "original_index": int(idx),
            "input": {
                "speech": speech,
            },
            "model_output": {
                "response": response,
            },
            "evaluation": {
                "scores": eval_result["scores"],
                "average_score": eval_result["average_score"],
                "gpt_thinking_and_analysis": eval_result["gpt_raw_output"],
            },
        }
        detailed_results.append(detailed_record)

        if verbose:
            print(f"Scores: {eval_result['scores']}")
            print(f"Average: {eval_result['average_score']:.2f}")

    # Save CSV results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nCSV results saved to {output_path}")

    # Save detailed JSON results
    if save_detailed_json:
        json_output_path = output_path.replace(".csv", "_detailed.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed JSON results saved to {json_output_path}")

    # Print summary
    print("\n=== Summary Statistics ===")
    score_cols = [
        "intent_understanding",
        "emotion_understanding",
        "pragmatic_inference",
        "proactive_fit",
        "conversational_naturalness",
        "average_score",
    ]
    print(results_df[score_cols].describe())

    return results_df


if __name__ == "__main__":
    # Configuration
    # evaluation_path =
    # output_path =
    test_num = 15  # Set to None for all records

    main(
        evaluation_path="/data/generation_results_with_labels.csv",
        output_path="/data/rubric_evaluation_predict_with_labels.csv",
        test_num=test_num,
        model_name="gpt-5",
        verbose=True,
        save_detailed_json=True,
    )

    main(
        evaluation_path="/data/evaluation_results_gpt.csv",
        output_path="/data/rubric_evaluation_predict_predicted.csv",
        test_num=test_num,
        model_name="gpt-5",
        verbose=True,
        save_detailed_json=True,
    )

    main(
        evaluation_path='/data/generation_results_with_labels.csv',
        output_path="/data/rubric_evaluation_predict_groundtruth.csv",
        test_num=test_num,
        model_name="gpt-5",
        verbose=True,
        save_detailed_json=True,
    )
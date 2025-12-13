import os
import pandas as pd
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import re

load_dotenv()


def get_speech(text: str) -> str:
    """Extract speech from the last <speech></speech> tags."""
    matches = re.findall(r"<speech>\s*(.*?)\s*</speech>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()  # Get the last match
    return text.strip()

def build_generation_prompt_with_labels(
        speech: str,
        intent: str,
        emotion: str,
        intents_definition: str,
        emotions_definition: str,
) -> str:
    """Build the generation prompt with intent/emotion labels."""
    prompt = f"""You are Pi. Your prime directive: Response based on the user's implicit emotion and intention.

Rules:
- First, you are given the intents and emotions aligned with the user's speech.
- Give your final supportive response.

This is the list of intent keywords you can use to describe the intent of the speech, with corresponding definitions:
<intent>
{intents_definition}
</intent>

This is the list of emotion keywords you can use to describe the emotion of the speech, with corresponding definitions:
<emotion>
{emotions_definition}
</emotion>

Format required:
<response>...</response>

Example:
<speech>Ugh, today was exhausting. I messed up the presentation.</speech>
<intent>self-disclosure, complaint</intent>
<emotion>frustration, disappointment</emotion>
<response>I hear how tough that was, it makes sense you feel drained.</response>

Now generate response for:
<speech>{speech}</speech>
<intent>{intent}</intent>
<emotion>{emotion}</emotion>"""

    return prompt


def generate_response_with_labels(
        llm: ChatOpenAI,
        speech: str,
        intent: str,
        emotion: str,
        intents_definition: str,
        emotions_definition: str,
        verbose: bool = False,
) -> str:
    """Generate response for a single record with intent/emotion labels."""
    prompt = build_generation_prompt_with_labels(
        speech, intent, emotion, intents_definition, emotions_definition
    )

    result = llm.invoke([("human", prompt)])
    response = result.content

    if verbose:
        print("\n" + "=" * 80)
        print("GENERATED RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80 + "\n")

    return response


def main(
        input_path: str,
        output_path: str,
        test_num: Optional[int] = 10,
        model_name: str = "gpt-4o",
        verbose: bool = True,
):
    """
    Main function to generate responses with intent/emotion labels.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the output CSV
        test_num: Number of records to process from top (None = all)
        model_name: OpenAI model name to use
        verbose: Whether to print generated responses
    """
    # Intent and emotion definitions
    intents_definition = """
- Greeting: Opening a conversation or acknowledging presence
- Question: Seeking information or clarification
- Action_Request: Asking for help or requesting an action
- Confirmation: Seeking validation or agreement
- Self_Disclosure: Sharing personal information or feelings
- Complaint: Expressing dissatisfaction or frustration
- Compliment: Expressing appreciation or praise
- Farewell: Ending a conversation
"""

    emotions_definition = """
- Neutral: Calm, matter-of-fact emotional state
- Happy: Joy, contentment, satisfaction
- Sad: Unhappiness, disappointment, grief
- Angry: Frustration, irritation, anger
- Anxious: Worry, nervousness, stress
- Confused: Uncertainty, bewilderment
- Excited: Enthusiasm, anticipation, eagerness
"""

    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records from {input_path}")

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

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Generating"):
        # Get speech, intent, emotion
        speech_raw = row.get("speech", "")
        speech = get_speech(speech_raw) if "<speech>" in speech_raw else speech_raw

        intent = row.get("groundtruth_intent", "")
        emotion = row.get("groundtruth_emotion", "")

        if verbose:
            print(f"\n{'#' * 80}")
            print(f"RECORD {idx}")
            print(f"{'#' * 80}")
            print(f"Speech: {speech[:200]}..." if len(speech) > 200 else f"Speech: {speech}")
            print(f"Intent: {intent}")
            print(f"Emotion: {emotion}")

        # Generate response
        generation = generate_response_with_labels(
            llm=llm,
            speech=speech,
            intent=intent,
            emotion=emotion,
            intents_definition=intents_definition,
            emotions_definition=emotions_definition,
            verbose=verbose,
        )

        # Build result record
        result = {
            "original_index": idx,
            "speech": f"<speech>{speech}</speech>",
            "groundtruth_intent": intent,
            "groundtruth_emotion": emotion,
            "generation": generation,
        }
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


if __name__ == "__main__":
    # Configuration
    input_path = "/data/evaluation_results_gpt.csv"
    output_path = "/data/generation_results_with_labels.csv"
    test_num = 10  # Set to None for all records

    main(
        input_path=input_path,
        output_path=output_path,
        test_num=test_num,
        model_name="gpt-5",
        verbose=True,
    )
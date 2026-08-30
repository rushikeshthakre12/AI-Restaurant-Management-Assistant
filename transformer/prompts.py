"""
Practical 8 (CO5) continued: Prompt Engineering -- Zero-shot, One-shot, and
Few-shot prompt construction for the response-generation step.

These functions build the actual prompt string that would be sent to a
generative model. build_response() below runs a template-based generator by
default (works with zero setup, no internet needed); call_huggingface_llm()
shows how to swap in a real Hugging Face pipeline call using this exact
prompt on a machine with internet access.
"""
from textwrap import dedent

TASK_INSTRUCTION = "Generate a short, polite restaurant assistant reply for the customer's request."


def zero_shot_prompt(customer_request: str) -> str:
    """No examples at all -- just the task instruction and the new input."""
    return dedent(f"""\
        {TASK_INSTRUCTION}

        Customer request: "{customer_request}"
        Reply:""")


def one_shot_prompt(customer_request: str) -> str:
    """One worked example, then the new input."""
    return dedent(f"""\
        {TASK_INSTRUCTION}

        Example:
        Customer request: "Book a table for 2 at 7 PM."
        Reply: "Your table for 2 has been booked for 7 PM. See you soon!"

        Customer request: "{customer_request}"
        Reply:""")


def few_shot_prompt(customer_request: str) -> str:
    """Several worked examples spanning different intents, then the new input."""
    return dedent(f"""\
        {TASK_INSTRUCTION}

        Example:
        Customer request: "Book a table for 2 at 7 PM."
        Reply: "Your table for 2 has been booked for 7 PM. See you soon!"

        Example:
        Customer request: "Cancel my reservation."
        Reply: "Your reservation has been cancelled. Let us know if you'd like to rebook."

        Example:
        Customer request: "Suggest something spicy under 300."
        Reply: "You might enjoy our Veg Manchurian or Chole Bhature -- both spicy and under 300."

        Customer request: "{customer_request}"
        Reply:""")


def call_huggingface_llm(prompt: str, model_name: str = "google/flan-t5-small") -> str:
    """
    Real Hugging Face inference. Requires internet access to download the
    model on first use (not available in this sandbox, which is why the
    demo below uses build_response()'s template generator instead) -- this
    function is written to be dropped in as-is on a machine with internet.
    """
    from transformers import pipeline
    generator = pipeline("text2text-generation", model=model_name)
    result = generator(prompt, max_new_tokens=40)
    return result[0]["generated_text"]


if __name__ == "__main__":
    request = "Book a table for 5 people tomorrow at 8 PM."
    print("=== ZERO-SHOT PROMPT ===")
    print(zero_shot_prompt(request))
    print("\n=== ONE-SHOT PROMPT ===")
    print(one_shot_prompt(request))
    print("\n=== FEW-SHOT PROMPT ===")
    print(few_shot_prompt(request))

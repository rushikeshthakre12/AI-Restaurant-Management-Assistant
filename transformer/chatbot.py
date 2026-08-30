"""
Final response generation step:
  ML Intent Classification + NLP Entity Extraction + Business Logic
  + Transformer-based Response Generation

Default generator is template-based (deterministic, zero setup, explainable
in a viva). If TRANSFORMERS_USE_LLM=1 is set in the environment AND the
machine has internet access, generate_reply() will instead call the real
Hugging Face pipeline via transformer.prompts.call_huggingface_llm() with a
few-shot prompt -- same interface either way.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformer.prompts import few_shot_prompt

_TEMPLATES = {
    "greeting": "Hello! Welcome to our restaurant. How can I help you today?",
    "table_booking": "Your table for {people} has been booked for {time} on {date}.",
    "cancel_booking": "Your booking has been cancelled. Let us know if you'd like to rebook anytime.",
    "modify_booking": "Your booking has been updated. Please check your booking history to confirm the new details.",
    "menu_query": "Here are some options from our menu that match what you're looking for.",
    "food_recommendation": "Based on what you're looking for, here are a few dishes you might enjoy.",
    "place_order": "Your order for {items} has been placed. We'll have it ready shortly!",
    "modify_order": "Your order has been updated.",
    "cancel_order": "Your order has been cancelled.",
    "restaurant_hours": "We're open daily from 11 AM to 11 PM.",
    "restaurant_location": "We're located at 12 MG Road, Nagpur. Look forward to seeing you!",
    "previous_order": "Here is your order history.",
    "payment_query": "We accept cash, cards, UPI, and digital wallets.",
    "complaint": "We're really sorry to hear that. Your feedback has been noted and we'll work on it right away.",
    "review": "Thank you so much for sharing your experience with us!",
    "goodbye": "Thank you for visiting! Have a wonderful day.",
    "unknown": "I'm sorry, I didn't quite understand that. Could you rephrase, or ask about booking, ordering, or our menu?",
}


def generate_reply(intent: str, entities: dict | None = None, extra: dict | None = None) -> str:
    entities = entities or {}
    extra = extra or {}

    if os.getenv("TRANSFORMERS_USE_LLM") == "1":
        from transformer.prompts import call_huggingface_llm
        prompt = few_shot_prompt(extra.get("raw_text", intent))
        return call_huggingface_llm(prompt)

    template = _TEMPLATES.get(intent, _TEMPLATES["unknown"])
    fill = {
        "people": entities.get("NUMBER_OF_PEOPLE") or extra.get("people", "your group"),
        "time": (entities.get("TIME") or ["your requested time"])[0] if entities.get("TIME") else "your requested time",
        "date": (entities.get("DATE") or ["the requested date"])[0] if entities.get("DATE") else "the requested date",
        "items": ", ".join(entities.get("FOOD_ITEM", [])) or extra.get("items", "your items"),
    }
    try:
        return template.format(**fill)
    except KeyError:
        return template


if __name__ == "__main__":
    import sys as _s
    from pathlib import Path as _P
    _s.path.append(str(_P(__file__).resolve().parent.parent))
    from nlp.ner import extract_entities
    from ml.predict_intent import predict_intent

    tests = [
        "book a table for 4 people tomorrow at 8 pm",
        "cancel my booking",
        "i want two paneer pizzas and one coke",
        "what are your restaurant timings",
    ]
    for t in tests:
        intent = predict_intent(t)["intent"]
        entities = extract_entities(t)
        reply = generate_reply(intent, entities)
        print(f"User: {t}\nIntent: {intent}  Entities: {entities}\nBot: {reply}\n")

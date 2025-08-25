"""
Generator: creates new creativity dataset items using LLM APIs.

- Prompts the model to produce (context, predictable, creative, meta_creative, surprise_source, domain)
- You can request items by domain/topic.
- Compatible with run_llm.py for batch processing
- Outputs results in JSON format
"""
import json
import random
import time
import argparse
from pathlib import Path
from typing import Dict
from src.data_creation import LLMInterface

SYSTEM = (
    "You are a dataset author creating compact, high-signal training examples "
    "for adversarial creativity."
)

TEMPLATE = """
You will create a single dataset item with the following JSON fields:
- id: a short unique string
- context: 1-2 sentences setting up a scenario in {topic}
- predictable: a continuation that a competent but conventional writer would produce
- creative: a surprising but coherent continuation that fits the context
- meta_creative: a reflection that generalizes the surprise into a principle/lens
- surprise_source: a compact label (e.g., 'scale_inversion', 'context_inversion')
- domain: a specific subdomain (e.g., 'renaissance_art', 'indian_classical')

Constraints:
- Keep each continuation to 1-3 sentences.
- Avoid brand-new factual claims that could be false
- Lean into style/structure/abstraction
- Ensure the creative continuation is genuinely non-trivial vs. predictable

Return ONLY valid JSON, no commentary.
""".strip()

DOMAINS = {
    "history": ["political_history","science_history","indian_history","modern_history"],
    "art": ["renaissance_art","digital_art","design","photography","indian_art"],
    "music": ["classical_music","jazz","indian_classical","rock","pop_music"],
    "poetry": ["japanese_poetry","urdu_poetry","modern_poetry","classical_poetry"],
    "fiction": ["sci_fi","fantasy","literary_fiction","crime_fiction","magical_realism"],
    "news": ["tech_business_news","policy_news","health_news","education_news","environment_news"],
    "science": ["genetics","quantum_computing","neuroscience","materials_science","astrophysics"],
    "sports": ["football","cricket","basketball","chess","athletics"],
}

def gen_item(llm: LLMInterface, topic: str, model: str = "gpt-4",
            temperature: float = 0.9) -> Dict:
    """
    Generate a single dataset item using the LLM.
    
    Args:
        llm: LLM interface instance
        topic: Topic area to generate for
        model: Model to use for generation
        temperature: Temperature for generation
    
    Returns:
        Dict containing the generated item with metadata
    """
    domain = random.choice(DOMAINS[topic])
    user = TEMPLATE.format(topic=topic.replace("_"," "))
    resp = llm.generate(
        system_prompt=SYSTEM,
        user_prompt=user + f"\nChoose domain='{domain}'.",
        model=model,
        temperature=temperature
    )
    text = resp.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Cleanup and retry for common formatting issues
        text = text.strip().strip("```json").strip("```").strip()
        data = json.loads(text)
        
    # Add metadata
    data["difficulty"] = random.choice(["easy","medium","hard"])
    data["tags"] = [topic, domain, "gen"]
    data["rubric"] = {
        "weights": {
            "novelty": 0.4,
            "coherence": 0.3,
            "specificity": 0.2,
            "voice": 0.1
        },
        "scales": {
            "novelty": "0-5",
            "coherence": "0-5",
            "specificity": "0-5",
            "voice": "0-5"
        },
        "expected_traits": [
            "Coherent with context",
            "Surprising yet defensible",
            "Concrete details where helpful",
            "Consistent, engaging voice"
        ]
    }
    
    # Initialize scores
    data["scores"] = {
        "novelty": None,
        "coherence": None,
        "specificity": None,
        "voice": None,
        "overall": None
    }
    
    data["eval_notes"] = ""
    return data

def main():
    """Main entry point for generating creativity dataset items."""
    parser = argparse.ArgumentParser(description="Generate creativity dataset items")
    parser.add_argument("--topic", choices=list(DOMAINS.keys()), required=True,
                       help="Topic area to generate items for")
    parser.add_argument("--n", type=int, default=5,
                       help="Number of items to generate")
    parser.add_argument("--out", default="generated_items.json",
                       help="Output JSON file path")
    parser.add_argument("--model", default="gpt-4",
                       help="Model to use for generation")
    parser.add_argument("--temperature", type=float, default=0.9,
                       help="Temperature for generation")
    args = parser.parse_args()

    # Create LLM interface
    llm = LLMInterface()
    
    # Generate items
    out = []
    for _ in range(args.n):
        out.append(gen_item(llm, args.topic, args.model, args.temperature))
        time.sleep(0.3)  # gentle pacing
        
    # Format output
    dataset = {
        "dataset_info": {
            "name": "Adversarial Creativity Training Dataset",
            "version": "2.0",
            "description": "Training examples for adversarial creativity across domains",
            "total_examples": len(out),
            "categories": list(DOMAINS.keys())
        },
        "items": out
    }
    
    # Save to file
    Path(args.out).write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"Wrote {args.n} items to {args.out}")

if __name__ == "__main__":
    main()

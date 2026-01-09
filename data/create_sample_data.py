"""
Create sample data mimicking the LLM Classification competition format.
This allows testing the pipeline without Kaggle credentials.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Sample prompts representing typical chatbot queries
SAMPLE_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "What are the health benefits of regular exercise?",
    "How do I make a classic Italian pasta carbonara?",
    "Explain quantum computing to a 10-year-old.",
    "What's the difference between SQL and NoSQL databases?",
    "Write a haiku about autumn leaves.",
    "How can I improve my public speaking skills?",
    "Explain the theory of relativity simply.",
    "What are the best practices for code review?",
    "How do neural networks learn?",
    "Write a short story about a robot learning to paint.",
    "What causes climate change?",
    "How do I start investing in stocks?",
    "Explain blockchain technology.",
    "What are the principles of good UI design?",
    "How does photosynthesis work?",
    "Write a poem about the ocean.",
    "What are effective study techniques?",
    "How do vaccines work?",
]

# Sample response templates (varying quality/style)
RESPONSE_TEMPLATES_GOOD = [
    "That's a great question! {topic} is fundamentally about {explanation}. Here's a simple way to think about it: {analogy}. In practice, this means {practical}.",
    "Let me break this down for you:\n\n1. **First**, {point1}\n2. **Second**, {point2}\n3. **Third**, {point3}\n\nIn summary, {summary}.",
    "I'd be happy to explain! {topic} can be understood as {explanation}. The key insight is that {insight}. This is important because {importance}.",
]

RESPONSE_TEMPLATES_MEDIUM = [
    "{topic} is {explanation}. It works by {mechanism}. Hope that helps!",
    "So basically, {topic} means {explanation}. People use it for {use_case}.",
    "Here's what you need to know about {topic}: {explanation}. The main thing is {main_point}.",
]

RESPONSE_TEMPLATES_SHORT = [
    "{topic} is {short_explanation}.",
    "It's basically {short_explanation}. Pretty simple really.",
    "{short_explanation}. That's the gist of it.",
]


def generate_response(template_type: str, topic: str) -> str:
    """Generate a response based on template type."""
    np.random.seed(hash(topic + template_type) % 2**32)

    fillers = {
        'topic': topic.lower(),
        'explanation': f"a way to {np.random.choice(['understand', 'approach', 'solve', 'think about'])} {topic.lower()}",
        'analogy': f"like {np.random.choice(['building blocks', 'a recipe', 'learning to ride a bike', 'solving a puzzle'])}",
        'practical': f"you can {np.random.choice(['apply this', 'use this knowledge', 'leverage this understanding'])} in everyday situations",
        'point1': f"understand the basics of {topic.lower()}",
        'point2': f"consider the key factors involved",
        'point3': f"apply what you've learned",
        'summary': f"mastering {topic.lower()} takes practice but is achievable",
        'insight': f"everything connects together systematically",
        'importance': f"it affects many aspects of our daily lives",
        'mechanism': f"following certain principles and patterns",
        'use_case': f"various practical applications",
        'main_point': f"consistency and understanding are key",
        'short_explanation': f"a fundamental concept in this domain",
    }

    if template_type == 'good':
        template = np.random.choice(RESPONSE_TEMPLATES_GOOD)
    elif template_type == 'medium':
        template = np.random.choice(RESPONSE_TEMPLATES_MEDIUM)
    else:
        template = np.random.choice(RESPONSE_TEMPLATES_SHORT)

    return template.format(**fillers)


def create_sample_data(n_train: int = 200, n_test: int = 50, seed: int = 42) -> tuple:
    """Create sample train and test datasets."""
    np.random.seed(seed)

    train_data = []
    for i in range(n_train):
        prompt = np.random.choice(SAMPLE_PROMPTS)
        topic = prompt.split()[0:3]
        topic = ' '.join(topic)

        # Randomly assign quality to each response
        quality_a = np.random.choice(['good', 'medium', 'short'], p=[0.4, 0.4, 0.2])
        quality_b = np.random.choice(['good', 'medium', 'short'], p=[0.4, 0.4, 0.2])

        response_a = generate_response(quality_a, topic)
        response_b = generate_response(quality_b, topic)

        # Determine winner based on quality (with some noise)
        quality_score = {'good': 2, 'medium': 1, 'short': 0}
        score_a = quality_score[quality_a] + np.random.normal(0, 0.5)
        score_b = quality_score[quality_b] + np.random.normal(0, 0.5)

        if abs(score_a - score_b) < 0.3:
            winner = 'tie'
        elif score_a > score_b:
            winner = 'model_a'
        else:
            winner = 'model_b'

        train_data.append({
            'id': f'train_{i:04d}',
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
            'winner_model_a': 1 if winner == 'model_a' else 0,
            'winner_model_b': 1 if winner == 'model_b' else 0,
            'winner_tie': 1 if winner == 'tie' else 0,
        })

    test_data = []
    for i in range(n_test):
        prompt = np.random.choice(SAMPLE_PROMPTS)
        topic = prompt.split()[0:3]
        topic = ' '.join(topic)

        quality_a = np.random.choice(['good', 'medium', 'short'], p=[0.4, 0.4, 0.2])
        quality_b = np.random.choice(['good', 'medium', 'short'], p=[0.4, 0.4, 0.2])

        response_a = generate_response(quality_a, topic)
        response_b = generate_response(quality_b, topic)

        test_data.append({
            'id': f'test_{i:04d}',
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
        })

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


if __name__ == '__main__':
    output_dir = Path(__file__).parent

    print("Creating sample data...")
    train_df, test_df = create_sample_data(n_train=200, n_test=50)

    # Save
    train_df.to_csv(output_dir / 'train.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nLabel distribution:")
    print(f"  winner_model_a: {train_df['winner_model_a'].sum()}")
    print(f"  winner_model_b: {train_df['winner_model_b'].sum()}")
    print(f"  winner_tie: {train_df['winner_tie'].sum()}")
    print(f"\nSaved to {output_dir}")

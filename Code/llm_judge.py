import matplotlib.pyplot as plt
from openai import OpenAI
import sys
import re

client = OpenAI(api_key='')  # Replace with your API key

# Weighting for different criteria
weights = {
    "Relevance": 0.2,
    "Correctness": 0.2,
    "Coherence": 0.2,
    "Conciseness": 0.2,
    "Consistency": 0.2
}

# Mapping from response file names to strategy names
strategy_name_mapping = {
    'responses_zs.txt': 'Zero Shot',
    'responses_os.txt': 'One Shot',
    'responses_fs.txt': 'Few Shot',
    'responses_cot.txt': 'Chain of Thought',
    'responses_rp.txt': 'Role Playing',
    'responses_ccot.txt': 'Contrastive CoT',
    'responses_sc.txt': 'Self Consistency'
}

# Function to evaluate a single response using GPT-4o-mini
def evaluate_response_with_gpt4omini(response):
    judge_prompt = f"""
    You are a highly critical evaluator. Rate each criterion from 1 to 10, with a high level of sensitivity. Only give a score of 10 if the response fully meets the highest standard of quality.

    - **Relevance**: Does the response directly address the query without deviating? Give a 10 only if it is completely on-topic.
    - **Correctness**: Are all facts presented accurate and well-supported? Rate strictly on factual accuracy.
    - **Coherence**: Is the response logically organized, with each part flowing smoothly? Reserve a 10 for perfectly structured answers.
    - **Conciseness**: Is the response free of unnecessary information while covering essential points? Give a 10 only if it’s succinct yet complete.
    - **Consistency**: Does the response maintain a consistent style, tone, and focus? Award a 10 only if the response fully avoids contradictions or shifts in focus.

    Respond in this exact format:

    Relevance: <numeric score>
    Correctness: <numeric score>
    Coherence: <numeric score>
    Conciseness: <numeric score>
    Consistency: <numeric score>

    Response: {response}
    """

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a highly critical judge evaluating text and will respond with numeric scores only."},
            {"role": "user", "content": judge_prompt}
        ],
    )
    response_text = chat_completion.choices[0].message.content
    print(f"Raw API Response for Query: {response_text}")  # Debugging: Print raw response from API
    return parse_scores(response_text)

# Function to parse scores returned by GPT-4o-mini
def parse_scores(text):
    scores = {"Relevance": 0, "Correctness": 0, "Coherence": 0, "Conciseness": 0, "Consistency": 0}
    for line in text.split("\n"):
        match = re.search(r"(Relevance|Correctness|Coherence|Conciseness|Consistency):\s*(\d+)", line)
        if match:
            criterion = match.group(1)
            score = int(match.group(2))
            scores[criterion] = score
        else:
            print(f"Warning: Could not parse line: {line}")  # Log unparsed lines for debugging
    print(f"Parsed Scores: {scores}")  # Debugging: Print parsed scores for each query
    return scores

# Function to apply weighted scoring
def calculate_weighted_score(scores):
    weighted_score = (
        scores["Relevance"] * weights["Relevance"] +
        scores["Correctness"] * weights["Correctness"] +
        scores["Coherence"] * weights["Coherence"] +
        scores["Conciseness"] * weights["Conciseness"] +
        scores["Consistency"] * weights["Consistency"]
    )
    print(f"Weighted Score: {weighted_score}")  # Debugging: Print weighted score for each response
    return weighted_score

# Function to load responses from response files
def load_responses(file_paths):
    responses = []
    for file_path in file_paths:
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
            response_list = re.split(r"\[\d+\] ", content)[1:]  # Split by pattern and ignore first empty match
            responses.append(response_list)
    return responses

# Main function to evaluate and select the best response for each query
def evaluate_responses_for_queries():
    response_files = [
        'responses_zs.txt', 'responses_os.txt', 'responses_fs.txt', 
        'responses_cot.txt', 'responses_rp.txt', 'responses_ccot.txt', 'responses_sc.txt'
    ]
    
    all_responses = load_responses(response_files)
    strategy_count = {file: 0 for file in response_files}
    total_scores = {file: 0 for file in response_files}
    response_counts = {file: 0 for file in response_files}

    # Loop through each query
    for i in range(len(all_responses[0])):
        print(f"\nEvaluating Query {i + 1} of {len(all_responses[0])}...")
        scores = []
        for response_list, response_file in zip(all_responses, response_files):
            response = response_list[i].strip()
            score = evaluate_response_with_gpt4omini(response)
            weighted_score = calculate_weighted_score(score)
            scores.append(weighted_score)
            total_scores[response_file] += weighted_score
            response_counts[response_file] += 1

        # Check for ties and use normalized scores for tie-breaking
        max_score = max(scores)
        best_indices = [index for index, score in enumerate(scores) if score == max_score]
        
        if len(best_indices) > 1:  # If there's a tie
            best_response_index = max(best_indices, key=lambda idx: total_scores[response_files[idx]] / response_counts[response_files[idx]])
        else:
            best_response_index = scores.index(max_score)

        best_prompt_strategy = response_files[best_response_index]
        strategy_count[best_prompt_strategy] += 1

    # Calculate normalized scores
    normalized_scores = {file: total_scores[file] / response_counts[file] for file in response_files}

    # Print normalized scores
    print("\nNormalized Scores for Each Prompt Strategy:")
    for file, score in normalized_scores.items():
        print(f"{strategy_name_mapping[file]}: {score:.2f}")

    # Generate graphs
    generate_strategy_charts(strategy_count, normalized_scores)

# Function to generate bar charts
def generate_strategy_charts(strategy_count, normalized_scores):
    # Bar chart for the count of best responses
    strategies = [strategy_name_mapping[file] for file in strategy_count.keys()]
    counts = list(strategy_count.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(strategies, counts)
    plt.xlabel("Number of Queries with Best Response")
    plt.ylabel("Prompt Strategy")
    plt.title("Best Prompt Strategy for Each Query")
    plt.show(block=True)  # Block execution until the window is closed

    # Bar chart for normalized scores
    normalized_strategies = [strategy_name_mapping[file] for file in normalized_scores.keys()]
    normalized_values = list(normalized_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(normalized_strategies, normalized_values)
    plt.xlabel("Prompt Strategy")
    plt.ylabel("Average Normalized Score")
    plt.title("Average Normalized Score for Each Prompt Strategy")
    plt.xticks(rotation=45)
    plt.show(block=True)  # Block execution until the window is closed

if __name__ == "__main__":
    evaluate_responses_for_queries()

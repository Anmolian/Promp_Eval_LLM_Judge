from openai import OpenAI
import sys
import os

# Manually set your OpenAI API key here
client = OpenAI(api_key='')

# Check if the prompt file is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <prompt_file>")
    sys.exit(1)

# Read the prompt from a text file provided as a command-line argument.
prompt_file = sys.argv[1]
with open(prompt_file, "r") as file:
    zero_shot_prompt = file.read()

# The TREC RAG queries file is fixed; set its path here.
query_file = "topics.rag24.test.txt"  # This remains fixed.

with open(query_file, "r") as file:
    queries = file.readlines()

# Initialize an empty list to store the responses.
responses = []

# Loop through each query to send it to the OpenAI API and get a response.
for index, query in enumerate(queries):
    query = query.strip()  # Remove any leading/trailing whitespace from the query.
    
    # Combine the zero-shot prompt with the current query to form the input for the API.
    prompt = f"{zero_shot_prompt}\nQuestion: {query}\nAnswer:"
    
    # Call the OpenAI API to generate a response based on the prompt.
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Use the desired model
        messages=[
            {"role": "system", "content": zero_shot_prompt.splitlines()[0]},  # Extracting the system part from your prompt
            {"role": "user", "content": prompt}
        ],
    )
    
    # Extract the generated response text from the API response.
    response_text = chat_completion.choices[0].message.content  # Access the content correctly
    responses.append(f"[{index + 1}] {response_text}")  # Format the response with numbering, without quotes.
    
    # Print the number of queries processed so far on the same line
    sys.stdout.write(f"\rProcessed query {index + 1} of {len(queries)}")
    sys.stdout.flush()

# Write all responses to an output file.
with open("responses_ccot.txt", "w", encoding='utf-8') as file: # Specify the output file name for different prompts
    for response in responses:
        file.write(response + "\n\n")  # Write each response followed by two empty lines in the output file.

print("\nVoilà")


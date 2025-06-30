import openai
import pandas as pd
import json
import time
from tqdm import tqdm

# === Set your OpenAI API key here ===
openai.api_key = "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"

# === Load data and prompts ===
df = pd.read_csv("dataset.csv")
with open("prompts.json", "r") as f:
    prompt_templates = json.load(f)

# === Store results ===
results = []

# === Loop through each prompt and each row ===
for i, prompt in enumerate(prompt_templates):
    print(f"\nRunning Prompt {i+1}/{len(prompt_templates)}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        expected = row['label']
        filled_prompt = prompt.replace("{text}", text)

        try:
            # Send request to OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": filled_prompt}
                ],
                temperature=0  # Make it deterministic
            )
            reply = response.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error: {e}")
            reply = "ERROR"

        # Save each result
        results.append({
            "text": text,
            "expected": expected,
            "prompt_version": i + 1,
            "prompt_template": prompt,
            "llm_response": reply
        })

        time.sleep(1.2)  # To avoid rate limit (optional)

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("results.csv", index=False)
print("\n✅ All prompt results saved to results.csv")

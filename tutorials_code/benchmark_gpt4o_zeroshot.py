import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Load API keys from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize OpenAI client
client = OpenAI(api_key=config["openai_api_key"])

# Sample financial questions
questions = [
    {
        "question": "Sammy Sneadle, CFA, is the founder of the Everglades Fund. How did he violate the standard by not disclosing back-tested data?",
        "choiceA": "He did not disclose the use of back-tested data.",
        "choiceB": "He failed to deduct fees before returns.",
        "choiceC": "He did not show a weighted composite of similar portfolios.",
        "answer": "A"
    },
    {
        "question": "What is the primary goal of corporate finance?",
        "choiceA": "Maximizing shareholder value",
        "choiceB": "Minimizing operational costs",
        "choiceC": "Increasing market share",
        "answer": "A"
    }
    # Add more questions as needed
]

def ask_openai(question, choiceA, choiceB, choiceC, model="gpt-4o"):
    """Generate a zero-shot response to a financial question"""
    system_prompt = (
        "You are a CFA (chartered financial analyst) taking a test. "
        "You will be given a question with three possible answers (A, B, and C). "
        "Provide only the letter for the correct choice (A, B, or C)."
    )
    
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"A. {choiceA}\n"
        f"B. {choiceB}\n"
        f"C. {choiceC}\n\n"
        "Which choice is correct? Answer with just the letter A, B, or C."
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

def evaluate_questions(questions_list, model="gpt-4o"):
    """Evaluate model on a list of questions"""
    results = []
    correct_count = 0
    
    for q in tqdm(questions_list, desc=f"Testing {model}"):
        question = q["question"]
        choiceA = q["choiceA"]
        choiceB = q["choiceB"] 
        choiceC = q["choiceC"]
        expected = q["answer"].upper().strip()
        
        # Get model response
        response = ask_openai(
            question=question,
            choiceA=choiceA,
            choiceB=choiceB,
            choiceC=choiceC,
            model=model
        )
        
        # Extract just the letter (A, B, or C)
        generated = response.upper().strip()
        if len(generated) > 1:
            # If response contains more than just the letter, try to extract the letter
            if "A" in generated:
                generated = "A"
            elif "B" in generated:
                generated = "B" 
            elif "C" in generated:
                generated = "C"
            else:
                generated = "INVALID"
        
        # Check if correct
        is_correct = generated == expected
        if is_correct:
            correct_count += 1
        
        # Store result
        results.append({
            "question": question[:50] + "..." if len(question) > 50 else question,
            "model_response": response,
            "generated": generated,
            "expected": expected,
            "correct": is_correct
        })
    
    # Calculate accuracy
    accuracy = correct_count / len(questions_list) if questions_list else 0
    
    return {
        "accuracy": accuracy,
        "results": results
    }

if __name__ == "__main__":
    # Run the evaluation
    evaluation = evaluate_questions(questions, model="gpt-4o")
    
    # Print results
    print(f"\nAccuracy: {evaluation['accuracy']:.2%}")
    
    print("\nResults:")
    for i, r in enumerate(evaluation['results']):
        print(f"Q{i+1}: {r['question']}")
        print(f"Response: {r['model_response']}")
        print(f"Expected: {r['expected']}, Got: {r['generated']}")
        print("✓" if r['correct'] else "✗")
        print()
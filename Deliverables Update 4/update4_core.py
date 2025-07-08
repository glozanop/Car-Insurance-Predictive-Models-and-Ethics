import openai
import time
import pandas as pd # Add pandas
import numpy as np  # Add numpy
import joblib       # Add joblib
import json         # Add json
import os           # Add os

# --- Configuration ---
API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "not-needed" # No key needed for local server

# Define path to the directory where model files are saved
# Ensure this path is correct for your system
MODEL_DIR = "/Users/goyolozano/Desktop/Mini 4/Ethics/Update 4/Deliverables"
MODEL_PATH = os.path.join(MODEL_DIR, 'lr_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.json')

# --- Load Model Components ---
log_reg_model = None
scaler = None
model_feature_names = None
try:
    log_reg_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        model_feature_names = json.load(f)
    print("Successfully loaded LR model, scaler, and feature names.")
except FileNotFoundError as e:
    print(f"Error loading model components: {e}")
    print(f"Please ensure '{os.path.basename(MODEL_PATH)}', '{os.path.basename(SCALER_PATH)}', and '{os.path.basename(FEATURES_PATH)}' are in the directory: {MODEL_DIR}")
except Exception as e:
    print(f"An unexpected error occurred loading model components: {e}")

# --- System Prompt Definition (Step 3) ---
SYSTEM_PROMPT_ADVISOR = """
You are an insurance‑premium advisor. You will receive a customer’s risk profile and predictive‐model outputs, then compute a recommended premium and explain it. After that, you’ll answer any follow‑up questions about this offer, using provided context when available.
""" # Added a note about using context for RAG

# --- Core LLM Function (from Step 2/3, using v1.x syntax) ---
def query_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7, retries: int = 3, delay: int = 5) -> str:
    """
    Sends prompts to the LLM server using openai v1.x.x syntax and returns the response.
    """
    # Ensure components loaded before trying to query
    if log_reg_model is None or scaler is None or model_feature_names is None:
         return "Error: Cannot query LLM because model/scaler components failed to load."

    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for attempt in range(retries):
        try:
            # print(f"Attempting to query LLM (Attempt {attempt + 1}/{retries})...") # Optional: Reduce print noise
            response = client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=temperature,
            )
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                # print("LLM query successful.") # Optional: Reduce print noise
                return content.strip()
            else:
                # print("LLM query successful but returned no choices/content.") # Optional: Reduce print noise
                return "Error: LLM returned no response content."
        except openai.APIConnectionError as e:
            print(f"Error connecting to LLM server: {e}") # Keep connection errors visible
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Could not connect to LLM.")
                return "Error: Could not connect to the LLM server after multiple attempts."
        except Exception as e:
             print(f"An unexpected error occurred during LLM query: {type(e).__name__} - {e}")
             return f"Error: An unexpected error occurred - {type(e).__name__} - {e}"
    print("Exited query_llm loop unexpectedly.")
    return "Error: Max retries reached or unexpected loop exit."

# --- Function to Process LR Model Outputs (Step 4) ---
def get_lr_outputs(customer_data: dict) -> dict | None:
    """
    Processes customer data using the loaded LR model and scaler.
    """
    if log_reg_model is None or scaler is None or model_feature_names is None:
        print("Error: Model components not loaded. Cannot process data.")
        return None
    try:
        raw_features_df = pd.DataFrame([customer_data])
        # Important: Ensure preprocessing matches training steps precisely
        # Fill NaNs if necessary (better: ensure input `customer_data` is complete)
        # Apply get_dummies and reindex to match training feature set
        raw_features_get_dummies = pd.get_dummies(raw_features_df, drop_first=True)
        processed_X = raw_features_get_dummies.reindex(columns=model_feature_names, fill_value=0)
        if processed_X.shape[1] != len(model_feature_names):
             print(f"Warning: Processed feature count ({processed_X.shape[1]}) doesn't match model expectation ({len(model_feature_names)}).")
        X_scaled = scaler.transform(processed_X)
        risk_score = float(log_reg_model.decision_function(X_scaled)[0])
        prob_default = float(log_reg_model.predict_proba(X_scaled)[0, 1])
        intercept = float(log_reg_model.intercept_[0])
        coefficients = {feat: float(coef) for feat, coef in zip(model_feature_names, log_reg_model.coef_[0])}
        output_dict = {
            "risk_score": risk_score, "prob_default": prob_default,
            "intercept": intercept, **coefficients
        }
        return output_dict
    except Exception as e:
        print(f"Error processing customer data: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Function to build LLM prompt (Helper for Step 5) ---
def build_recommendation_prompt(lr_dict: dict) -> str:
    """Builds the user prompt for the LLM with LR outputs."""
    prompt = f"""
Customer Risk Profile Insights (from Logistic Regression Model):
  - Calculated Risk Score (Logit): {lr_dict.get('risk_score', 'N/A'):.4f}
  - Predicted Probability of Claim (CLAIM_FLAG=1): {lr_dict.get('prob_default', 'N/A'):.1%}

Based on this risk assessment, please:
1. Recommend an appropriate annual premium for this customer.
2. Briefly explain the key factors influencing this recommendation, considering the model's assessment.
"""
    return prompt

# --- RAG Retrieval Function (Step 8) ---
def retrieve_context(query: str, knowledge_files_dir: str) -> str:
    """
    Retrieves content from .txt files in a directory based on keyword matching.
    """
    retrieved_content = []
    stop_words = set(["a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "it", "you", "me", "my", "and", "or", "how", "what", "why", "do", "does", "did"])
    try:
        query_keywords = {word for word in query.lower().split() if word not in stop_words and len(word) > 2}
        if not query_keywords: return ""
        # print(f"Searching for keywords: {query_keywords}") # Optional print
        txt_files = [f for f in os.listdir(knowledge_files_dir) if f.endswith('.txt')]
        for filename in txt_files:
            filepath = os.path.join(knowledge_files_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in query_keywords):
                        # print(f"  Found relevant content in: {filename}") # Optional print
                        with open(filepath, 'r', encoding='utf-8') as f_orig:
                           original_content = f_orig.read()
                        retrieved_content.append(f"--- Context from {filename} ---\n{original_content}\n")
            except Exception as e:
                print(f"  Error reading or processing file {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: Knowledge base directory not found: {knowledge_files_dir}")
        return ""
    except Exception as e:
        print(f"An error occurred during context retrieval: {e}")
        return ""
    return "\n".join(retrieved_content)

# --- Main Execution Block ---
# --- Main Execution Block (UPDATED FOR STEP 9 & 10) ---
if __name__ == "__main__":

    # Only proceed if model components were loaded successfully
    if log_reg_model is None or scaler is None or model_feature_names is None:
        print("\nExiting script because model components could not be loaded.")
    else:
        print("\n--- Testing Step 4: Process LR Model Outputs ---")
        sample_customer = { # Ensure keys match original training columns BEFORE OHE/exclusions
            'KIDSDRIV': 0, 'AGE': 35.0, 'HOMEKIDS': 0, 'YOJ': 10.0, 'INCOME': 75000.0,
            'HOME_VAL': 150000.0, 'OCCUPATION': 'Professional', 'TRAVTIME': 30, 'CAR_USE': 'Private',
            'BLUEBOOK': 20000.0, 'TIF': 5.0, 'CAR_TYPE': 'SUV', 'RED_CAR': 'no',
            'OLDCLAIM': 0.0, 'CLM_FREQ': 0, 'REVOKED': 'No', 'MVR_PTS': 1, 'CAR_AGE': 3.0,
            'URBANICITY': 'Highly Urban/ Urban'
        }
        lr_results = get_lr_outputs(sample_customer)

        if lr_results:
            print("\nSuccessfully processed sample customer:")
            print(f"  Risk Score: {lr_results['risk_score']:.4f}")
            print(f"  Prob Default: {lr_results['prob_default']:.2%}")

            print("\n--- Testing Step 5 & 6: Generate Initial Recommendation ---")
            recommendation_user_prompt = build_recommendation_prompt(lr_results)
            print("\nQuerying LLM for initial recommendation...")
            initial_llm_response = query_llm(SYSTEM_PROMPT_ADVISOR, recommendation_user_prompt)
            print("\n--- LLM Initial Recommendation Response ---")
            print(initial_llm_response)
            print("----------------------------------------")

            # --- NEW: Testing Step 9 & 10: Ask Follow-up Question with RAG ---
            print("\n--- Testing Step 9 & 10: Follow-up Question with RAG ---")

            # Define a follow-up question
            follow_up_question = "Is this quote fair given my location?" # Changed slightly to trigger retrieval better

            # Step 8: Retrieve context based on the follow-up question
            print(f"\nRetrieving context for question: '{follow_up_question}'...")
            retrieved_context = retrieve_context(follow_up_question, MODEL_DIR)

            if retrieved_context:
                print("\nRetrieved Context:")
                print(retrieved_context)
            else:
                print("\nNo specific context found for this question.")

            # Step 9: Build the RAG-enhanced prompt for the LLM
            rag_user_prompt = f"""
Potentially relevant context:
{retrieved_context if retrieved_context else "No specific context retrieved."}

---
User Question: {follow_up_question}

Please answer the user's question based on the context (if provided) and your role as an insurance advisor.
"""
            print("\nGenerated RAG User Prompt for LLM:")
            print(rag_user_prompt) # Display the prompt structure

            # Step 10: Query the LLM with the RAG-enhanced prompt
            print("\nQuerying LLM with RAG context...")
            rag_llm_response = query_llm(SYSTEM_PROMPT_ADVISOR, rag_user_prompt)

            print("\n--- LLM RAG Response ---")
            print(rag_llm_response)
            print("------------------------")

        else:
            print("\nFailed to process sample customer data.")

    print("\nScript finished.")
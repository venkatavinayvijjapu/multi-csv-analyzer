import os
import sys

# Add project root to sys.path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from session_store import set_file, get_session
from tests.eval_framework import run_eval, make_standard_questions

def run_automated_eval():
    print("Setting up test session for evaluation...")
    session_id = "eval_test_session"
    
    # Create a dummy dataset
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "salary": [50000, 60000, 75000, 45000, 80000],
        "dept": ["HR", "Engineering", "Engineering", "Sales", "HR"]
    })
    
    # Upload to session
    set_file(session_id, "employees.csv", df)
    
    print("Running evaluation framework...")
    questions = make_standard_questions("employees.csv", "salary", "dept")
    
    # Run the evaluation
    report = run_eval(session_id, questions)
    
    # Print the result
    print(report.summary())

if __name__ == "__main__":
    run_automated_eval()

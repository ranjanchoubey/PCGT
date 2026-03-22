import os

base_dir = "research_project"

files_content = {
    "01_problem.md": """# Problem Statement

## Task for Copilot:
Expand this section with:

- Clear problem definition
- Why this problem matters
- Real-world applications
- What is missing in current solutions
- Your hypothesis

Write in detailed teaching style.

---

## Notes:
""",

    "02_background.md": """# Background & Fundamentals

## Task for Copilot:
Explain all required concepts:

- Core theory
- Definitions
- Prior knowledge needed
- Simple explanations + examples

Avoid assuming prior knowledge.

---

## Notes:
""",

    "03_related_work.md": """# Related Work

## Task for Copilot:
- Summarize existing approaches
- Compare methods
- Identify gaps
- Explain limitations of current work

---

## Notes:
""",

    "04_method.md": """# Proposed Method

## Task for Copilot:
- Step-by-step explanation
- Intuition behind approach
- Algorithm / workflow
- Why this works

Include examples.

---

## Notes:
""",

    "05_design_choices.md": """# Design Choices

## Task for Copilot:
Explain:

- Why each decision was made
- Alternatives considered
- Trade-offs
- Pros and cons

---

## Notes:
""",

    "06_experiments_plan.md": """# Experiment Plan

## Task for Copilot:
- What experiments to run
- Expected results
- Metrics to evaluate
- Variables to test

---

## Notes:
""",

    "07_results_raw.md": """# Raw Results

## Task for Copilot:
- Record outputs
- Observations
- Unexpected behaviors

DO NOT summarize, just log.

---

## Notes:
""",

    "08_analysis.md": """# Analysis

## Task for Copilot:
- Interpret results
- Why results happened
- Compare with expectations
- Insights

---

## Notes:
""",

    "09_limitations.md": """# Limitations

## Task for Copilot:
- Weaknesses
- Failure cases
- Assumptions that break

Be critical.

---

## Notes:
""",

    "10_future_work.md": """# Future Work

## Task for Copilot:
- Improvements
- Extensions
- New ideas

---

## Notes:
""",

    "11_experiment_logs.md": """# Experiment Logs

## Task for Copilot:
For each run include:

- What was changed
- Why
- Result
- Observation

---

## Logs:
""",

    "12_questions.md": """# Questions & Doubts

## Task:
- Write unclear points
- Things to investigate
- Confusions

---

## Questions:
""",

    "13_assumptions.md": """# Assumptions

## Task for Copilot:
List all assumptions:

- Data assumptions
- Model assumptions
- Constraints

---

## Notes:
""",

    "14_ideas_dump.md": """# Ideas Dump

## Task:
- Random ideas
- Future experiments
- Improvements

---

## Notes:
"""
}

# Create directories
os.makedirs(base_dir, exist_ok=True)

for filename, content in files_content.items():
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "w") as f:
        f.write(content)

print(f"✅ Research structure created in '{base_dir}'")
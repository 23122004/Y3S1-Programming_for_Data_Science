# Introduction
This dataset provides a deep look into AI-assisted software development, focusing on how coding hours, AI usage, stress levels, sleep duration, cognitive load, bugs, errors, and commits influence a developer’s overall productivity.

In 2025, AI became an essential part of modern software engineering—and understanding human + AI performance is now more important than ever.
This dataset helps researchers, ML practitioners, and engineering managers analyze how human behavior + AI tools shape coding outcomes.

# Why This Dataset is Important
AI adoption is increasing in software engineering.

Developer burnout and cognitive overload are major global concerns.

Productivity analytics are trending across tech companies.

Kaggle community loves behavioral and performance datasets.

Ideal for ML models related to human performance & work efficiency.

This dataset fills a crucial research gap by combining behavioral, cognitive, productivity, and AI-usage metrics.

# Dataset Summary
This dataset includes 1000 clean, complete rows and 13 structured features describing AI-assisted software developer performance. The variables are grouped into five categories:

**Productivity:** Hours_Coding, Lines_of_Code, Commits, Task_Duration_Hours

**Quality & Errors:** Bugs_Found, Bugs_Fixed, Errors

**AI Usage:** AI_Usage_Hours

**Human Condition:** Sleep_Hours, Coffee_Intake, Stress_Level, Cognitive_Load

**Performance Outcome:** Task_Success_R

# Column Description

| Column               | Description                          |
|----------------------|--------------------------------------|
| Hours_Coding         | Daily hours spent coding             |
| Lines_of_Code        | Total lines of code written          |
| Bugs_Found           | Number of bugs identified            |
| Bugs_Fixed           | Number of bugs fixed                 |
| AI_Usage_Hours       | Hours spent using AI coding tools    |
| Sleep_Hours          | Average daily sleep                  |
| Cognitive_Load       | Mental pressure score (0–100)        |
| Task_Success_Rate    | Completion score (0–100)             |
| Coffee_Intake        | Daily cups of coffee                 |
| Stress_Level         | Stress score (0–100)                 |
| Task_Duration_Hours  | Hours taken to complete a task       |
| Commits              | Total Git commits                    |
| Errors               | Number of coding errors              |


# Key Insights
Insight 1: AI Usage reduces errors & increases productivity

More AI_Usage_Hours → fewer Errors and more Bugs_Fixed.

Insight 2: Stress strongly affects performance

High Stress_Level → low Task_Success_Rate and higher Errors.

Insight 3: Sleep improves success rate

Higher Sleep_Hours → lower Cognitive_Load and higher Task_Success_Rate.

Insight 4: Coding hours vs cognitive overload

More Hours_Coding → higher Cognitive_Load and more Stress_Level.

Insight 5: Coffee increases productivity but spikes stress

Coffee_Intake ↑ → short-term boost but long-term stress ↑.

# Who Should Use This Dataset
Machine Learning Practitioners

AI Researchers

HR Analytics Teams

Engineering Managers

Productivity Scientists

Behavioral Psychologists

Data Scientists

# Possible ML Projects
Developer Productivity Prediction Model
- Predict Task_Success_Rate using coding + behavioral metrics.

Burnout Detection System
- Stress_Level + Sleep_Hours → burnout prediction.

AI Usage Optimization Model
- Determine ideal AI_Usage_Hours for best performance.

Cognitive Load Estimator
- Predict mental load based on workload metrics.

Coffee vs Productivity Analysis
- Quantify real effect of coffee on coding performance.

Error Rate Prediction Model
- Predict Errors using AI usage + stress + coding hours.

Software Engineering Efficiency Research
- Which combination of habits makes developers high-performing?
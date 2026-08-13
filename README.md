# schrade_projects

Data science projects and coursework from my Master's program. The larger, self-contained projects are listed first; the remaining folders collect smaller course assignments.

## Main projects

### [Master_Thesis/simulation](Master_Thesis/simulation/)
Multi-agent deliberation simulation for my thesis: a hybrid of LLM agents and Social Feedback Theory (SFT) modeling opinion dynamics in structured social networks. Agents with survey-based personas (GSS 2024) interact on an SBM graph, with Q-learning governing opinion expression; runs on local LLMs via Ollama. Includes network and pairwise simulation modes, a sweep runner, and extensive planning/runbook documentation.

### [Deep_Learning](Deep_Learning/)
Assignments and final project for a deep learning course.
- **Final project:** predicting income inequality (GINI coefficients) for countries with missing data using country similarity networks and graph neural networks, with Optuna hyperparameter tuning (see `final_project/DLSS_Project_Report.pdf`).
- **Assignments 1–4:** tabular learning on job vacancy data, satellite image classification (EuroSAT), graph learning on the Twitch gamers network, and further tasks — each with notebook and PDF write-up.

### [Statistical_Learning](Statistical_Learning/)
Final project for the Statistical Learning course (Winter 25/26): predicting FIFA player potential from age, physical, and cognitive attributes. Compares a neural network against lasso regression with nested cross-validation, evaluated via MSE/MAE (Quarto report in `report.pdf`).

### [Social_Media_Data_Project](Social_Media_Data_Project/)
End-to-end social media data analysis project: data collection, LLM-assisted labeling, and statistical analysis (bootstrap comparisons, engagement distributions) of YouTube video and user activity, with a full report (`SMDA_project.pdf`) and visualizations.

## Course assignments

- **[General_Data_Assignments](General_Data_Assignments/)** — four assignments from "Introduction to Computation for the Social Sciences": election prediction data, time series, tweet text processing, and SQL/database queries.
- **[Sentiment_Analysis](Sentiment_Analysis/)** — sentiment and emotion classification on Twitter/YouTube comments and the ISEAR dataset, including manual and model-based labeling.
- **[Social_Impact_Analysis_Twitter_YouTube](Social_Impact_Analysis_Twitter_YouTube/)** — analysis of US Congress tweets and YouTube channel data.
- **[Trends_Data_Analysis](Trends_Data_Analysis/)** — comparing Google Trends search interest with CDC ILINet influenza surveillance data.

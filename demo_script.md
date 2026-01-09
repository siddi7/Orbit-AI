# Demo Video Script (2-3 Minutes)

**00:00–00:20 — Introduction**
"Hi, I'm presenting ORBIT-Data Analyst. Data science usually requires hours of boilerplate code for cleaning, modeling, and validating—especially on small, high-stakes datasets. We built ORBIT to automate this entire pipeline, delivering actionable insights and explainable models in seconds."

**00:20–00:40 — Data Input**
"Let's look at a demo. I’m uploading a small dataset of job market trends. You can see the preview here—just 35 rows. Usually, this is too small for deep learning but perfect for robust statistical modeling."

**00:40–01:00 — EDA**
"The system immediately auto-generates Exploratory Data Analysis. Here's the distribution of our target variable, 'Demand', and a correlation matrix showing how Salary and Count interact with it."

**01:00–01:30 — Auto-Analysis**
"I click 'Run AutoAnalyst'. Behind the scenes, ORBIT is training five different models—including Ridge, Lasso, and Random Forest. Crucially, because the dataset is small (n=35), it automatically switches to Leave-One-Out Cross-Validation to ensure the results are statistically valid and not just overfitting."

**01:30–01:50 — Results & Diagnostics**
"The results are in. Random Forest performed best with an R-squared of 0.85. We can inspect the 'Predicted vs Actual' plot to see how well it tracks the trend, and check the residuals for any bias."

**01:50–02:15 — Explainability**
"For judges who need 'why', we have integrated SHAP values. This summary plot shows exactly which features—like 'Avg_Salary_log'—drove the predictions the most. It's not a black box."

**02:15–02:40 — Reporting**
"Finally, ORBIT generates a downloadable Executive Report in Markdown. It summarizes the findings, metrics, and recommendations, ready to be shared with stakeholders immediately. All artifacts are saved to the 'outputs' folder for reproducibility."

**02:40–03:00 — Conclusion**
"ORBIT-Data Analyst turns raw data into rigorous, explainable intelligence in under a minute. Thank you."

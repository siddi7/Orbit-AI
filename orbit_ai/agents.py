import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import re
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from .core import BaseAgent, SharedMemory, logger

class PlannerAgent(BaseAgent):
    def __init__(self, model="llama3"):
        super().__init__("Planner", "Project Manager", model)

    def run(self, memory: SharedMemory):
        goal = memory.context.get("goal", "")
        memory.log(self.name, f"Planning for goal: {goal}")
        
        prompt = f"""
        Goal: {goal}
        
        Break this goal into 4-6 distinct, high-level sequential steps for an autonomous data science team.
        Return ONLY a JSON array of strings. Example: ["Step 1...", "Step 2..."]
        """
        response = self.ask_llm(prompt, json_mode=True)
        try:
            # Clean up response if it contains markdown code blocks
            clean_response = response.replace("```json", "").replace("```", "").strip()
            # Find the list part
            match = re.search(r'\[.*\]', clean_response, re.DOTALL)
            if match:
                steps = json.loads(match.group())
            else:
                steps = [clean_response] # Fallback
            
            memory.plan = steps
            memory.log(self.name, f"Generated plan with {len(steps)} steps.")
        except Exception as e:
            memory.log(self.name, f"Failed to parse plan: {e}. Using default.")
            memory.plan = ["Load Data", "Analyze Data", "Build Model", "Report"]

class DataScoutAgent(BaseAgent):
    def __init__(self, model="llama3"):
        super().__init__("DataScout", "Data Engineer", model)

    def run(self, memory: SharedMemory):
        memory.log(self.name, "Searching for data...")
        # For this hackathon/demo, we will simulate finding a relevant dataset regarding the goal
        # or load a local CSV if mentioned.
        
        goal = memory.context.get("goal", "").lower()
        
        # Check if user provided a path in goal (simple heuristic)
        # In a real app, this would be more robust
        
        # Synthetic Generation Logic
        if "job" in goal or "career" in goal:
            df = self.generate_job_data()
            memory.data["main"] = df
            memory.log(self.name, "Generated synthetic Job Market dataset.")
        elif "sales" in goal or "revenue" in goal:
            df = self.generate_sales_data()
            memory.data["main"] = df
            memory.log(self.name, "Generated synthetic Sales dataset.")
        else:
            # Default generic dataset
            df = self.generate_generic_data()
            memory.data["main"] = df
            memory.log(self.name, "Generated generic synthetic dataset.")
            
        memory.insights.append(f"Data Loaded: {len(df)} rows, {len(df.columns)} columns.")

    def generate_job_data(self):
        years = np.arange(2020, 2027)
        roles = ["AI Engineer", "Data Scientist", "Software Dev", "Product Manager", "UX Designer"]
        data = []
        for y in years:
            for r in roles:
                demand = np.random.randint(1000, 5000) * (1 + (y-2020)*0.1) # Upward trend
                salary = np.random.randint(80000, 150000) * (1 + (y-2020)*0.05)
                data.append({"Year": y, "Role": r, "Demand": int(demand), "Avg_Salary": int(salary)})
        return pd.DataFrame(data)

    def generate_sales_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="W")
        sales = np.linspace(1000, 5000, 100) + np.random.normal(0, 500, 100)
        return pd.DataFrame({"Date": dates, "Sales": sales, "Marketing_Spend": sales * 0.2 + np.random.normal(0, 100, 100)})

    def generate_generic_data(self):
        return pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

class DataScienceAgent(BaseAgent):
    def __init__(self, model="llama3"):
        super().__init__("DataScientist", "Analyst", model)

    def run(self, memory: SharedMemory):
        df = memory.data.get("main")
        if df is None:
            memory.log(self.name, "No data found.")
            return

        memory.log(self.name, "Performing EDA...")
        
        # Describe
        desc = df.describe().to_string()
        memory.context["data_description"] = desc
        
        # Correlation (numeric only)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            memory.context["correlation"] = corr.to_string()
            
            # Create Heatmap
            fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
            memory.figures["correlation_heatmap"] = fig
            memory.log(self.name, "Generated Correlation Heatmap.")
        
        # Identifying Trends
        # Ask LLM for what to analyze based on columns
        columns = list(df.columns)
        prompt = f"Given columns {columns}, what is the most interesting trend to visualize? Return just the column names."
        # Simple heuristic for now: Plot first numeric vs time or first categorical
        
        memory.insights.append(f"Data Distribution:\\n{desc}")

class MLAgent(BaseAgent):
    def __init__(self, model="llama3"):
        super().__init__("MLAgent", "Machine Learning Engineer", model)

    def run(self, memory: SharedMemory):
        df = memory.data.get("main")
        if df is None: return

        # Simple Regression Logic
        # Try to predict the last numeric column based on others
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            memory.log(self.name, "Not enough numeric columns for regression.")
            return

        target_col = numeric_df.columns[-1]
        feature_cols = numeric_df.columns[:-1]
        
        X = numeric_df[feature_cols]
        y = numeric_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        memory.metrics = {"MSE": mse, "R2": r2}
        memory.insights.append(f"ML Model (Linear Regression) trained on '{target_col}'. R2 Score: {r2:.2f}")
        
        # Visualizing Actual vs Predicted
        fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title=f"Actual vs Predicted ({target_col})")
        fig.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
        memory.figures["prediction_plot"] = fig

        # Future Forecast (if Year exists)
        if "Year" in df.columns and target_col != "Year":
            future_years = np.array([[2027], [2028], [2029]])
            # Assuming simple mapping for demo if Year is a feature
            if "Year" in feature_cols:
                # We need to construct a dataframe for prediction matching X features
                # This is tricky autonomously. Simplification:
                pass

class CriticAgent(BaseAgent):
    def __init__(self, model="llama3"):
        super().__init__("Critic", "Reviewer", model)

    def run(self, memory: SharedMemory):
        insights = "\\n".join(memory.insights)
        prompt = f"""
        Review these insights:
        {insights}
        
        Identify 1 potential bias or limitation. Be brief.
        """
        critique = self.ask_llm(prompt)
        memory.insights.append(f"CRITIQUE: {critique}")

class SynthesizerAgent(BaseAgent):
    def __init__(self, model="llama3"):
        super().__init__("Synthesizer", "Technical Writer", model)

    def run(self, memory: SharedMemory):
        memory.log(self.name, "Drafting final report...")
        data = memory.to_json()
        
        prompt = f"""
        Write a professional Executive Summary based on this analysis context:
        {str(memory.insights)}
        
        Format as Markdown. Include key findings and a recommendation.
        """
        report = self.ask_llm(prompt)
        memory.reports["executive_summary"] = report
        memory.log(self.name, "Report generated.")

import pandas as pd
from transformers import pipeline

class BaseAgent:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.context = ""

    def add_to_context(self, text):
        self.context += f"\n{text}"

    def get_context(self):
        return self.context

    def execute(self, task, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

class MLAnalysisAgent(BaseAgent):
    def __init__(self, name="MLAnalysisAgent"):
        super().__init__(name, "Performs machine learning analysis and modeling.")

    def execute(self, task, **kwargs):
        if task == "train_model":
            return self.train_model(kwargs.get("data"), kwargs.get("target_var"), kwargs.get("feature_vars"))
        else:
            raise ValueError(f"Unknown task: {task}")

    def train_model(self, data, target_var, feature_vars):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        X = data[feature_vars].fillna(0)
        y = data[target_var].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        self.add_to_context(f"Model trained with MSE: {mse}")
        return model, mse

class PolicyAgent(BaseAgent):
    def __init__(self, name="PolicyAgent"):
        super().__init__(name, "Generates policy suggestions based on data analysis.")

    def execute(self, task, **kwargs):
        if task == "suggest_policy":
            return self.suggest_policy(kwargs.get("analysis_results"))
        else:
            raise ValueError(f"Unknown task: {task}")

    def suggest_policy(self, analysis_results):
        """
        Generate policy suggestions based on the analysis results.
        """
        policies = []
        for key, value in analysis_results.items():
            if value > 0.5:
                policies.append(f"Increase investment in {key} due to strong positive correlation with target outcomes.")
            else:
                policies.append(f"Consider deprioritizing {key} as its impact on target outcomes is minimal.")
        
        self.add_to_context(f"Policy Suggestions: {policies}")
        return policies

class SimulationAgent(BaseAgent):
    def __init__(self, name="SimulationAgent"):
        super().__init__(name, "Runs simulations to predict the impact of policy changes.")

    def execute(self, task, **kwargs):
        if task == "run_simulation":
            return self.run_simulation(kwargs.get("data"), kwargs.get("feature_changes"), kwargs.get("target_var"), kwargs.get("feature_vars"))
        else:
            raise ValueError(f"Unknown task: {task}")

    def run_simulation(self, data, feature_changes, target_var, feature_vars):
        """
        Simulate the impact of changes in feature values on the target variable.
        """
        simulated_data = data.copy()
        for feature, change in feature_changes.items():
            if feature in simulated_data.columns:
                simulated_data[feature] += change

        from sklearn.linear_model import LinearRegression

        X = simulated_data[feature_vars].fillna(0)
        y = simulated_data[target_var].fillna(0)

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        self.add_to_context(f"Simulation Results: Predictions generated for feature changes {feature_changes}")
        return predictions

class InteractiveSystem:
    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.name] = agent

    def interact(self, agent_name, task, **kwargs):
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found.")
        return agent.execute(task, **kwargs)

# Example Usage
if __name__ == "__main__":
    # Load your data
    data = pd.read_csv("your_data.csv")

    # Create system and agents
    system = InteractiveSystem()

    ml_analysis_agent = MLAnalysisAgent()
    policy_agent = PolicyAgent()
    simulation_agent = SimulationAgent()

    system.add_agent(ml_analysis_agent)
    system.add_agent(policy_agent)
    system.add_agent(simulation_agent)

    # Interact with MLAnalysisAgent
    model, mse = system.interact("MLAnalysisAgent", "train_model", data=data, target_var="target", feature_vars=["feature1", "feature2"])
    print("Trained Model MSE:", mse)

    # Interact with PolicyAgent
    analysis_results = {"feature1": 0.6, "feature2": 0.3}
    policies = system.interact("PolicyAgent", "suggest_policy", analysis_results=analysis_results)
    print("Policy Suggestions:", policies)

    # Interact with SimulationAgent
    feature_changes = {"feature1": 10, "feature2": -5}
    predictions = system.interact("SimulationAgent", "run_simulation", data=data, feature_changes=feature_changes, target_var="target", feature_vars=["feature1", "feature2"])
    print("Simulation Predictions:", predictions)

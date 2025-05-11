import pandas as pd
data = pd.read_csv("road3.csv")
auto_data = data.dropna(subset=["Pred_Gear"])
print("Pred_Gear distribution:")
print(auto_data["Pred_Gear"].value_counts())
print(f"Gear Accuracy: {(auto_data['Gear_output'] == auto_data['Pred_Gear']).mean():.4f}")
print("RPM and SpeedX stats when Pred_Gear == 1:")
print(auto_data[auto_data["Pred_Gear"] == 1][["RPM", "SpeedX"]].describe())
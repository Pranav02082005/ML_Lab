import pandas as pd

# Custom dataset: 4 attributes -> Sky, Temperature, Humidity, Wind => PlayTennis
data = {
    "Sky":         ["Sunny","Sunny","Rainy","Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity":    ["Normal","High","High","Normal"],
    "Wind":        ["Strong","Strong","Strong","Weak"],
    "PlayTennis":  ["Yes",  "Yes",  "No",   "Yes"]
}
df = pd.DataFrame(data)
attributes = ["Sky","Temperature","Humidity","Wind"]

def find_s(df, target_col, pos_class="Yes"):
    # Initialize hypothesis as most specific (first positive example)
    positives = df[df[target_col] == pos_class][attributes].values
    hypothesis = list(positives[0])
    
    for sample in positives[1:]:
        for i, val in enumerate(sample):
            if hypothesis[i] != val:
                hypothesis[i] = "?"   # Generalize
    return hypothesis

h = find_s(df, "PlayTennis")
print("Training Data:")
print(df)
print("\nMost Specific Hypothesis (FIND-S):")
print(h)

import json
import os

# Path to the JSON file
# json_file_path = r"D:\COLLEGE\Project\QR-Project\initial_stock.json"
json_file_path = r"D:\COLLEGE\Project\QR-Project\sold_stock.json"

# Function to count models in the JSON file

def count_models_in_json(json_file_path):
    model_count = {}

    # Check if the JSON file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            try:
                data = json.load(json_file)  # Load the existing data
                # Iterate through the list of entries
                for item in data:
                    model = item.get("Model")
                    if model:  # Check if the model exists
                        if model in model_count:
                            model_count[model] += 1
                        else:
                            model_count[model] = 1
            except json.JSONDecodeError:
                print("Failed to decode JSON. The file may be corrupted or empty.")
    else:
        print("JSON file does not exist.")

    return model_count

# Run the counting function
model_counts = count_models_in_json(json_file_path)

# Print the counts
print("Model counts:", model_counts)

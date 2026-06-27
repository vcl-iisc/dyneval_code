import json
import sys

def transform(data):
    """Transforms input JSON to {id: id.png} mapping."""
    if isinstance(data, dict):
        # Single item
        return {data["id"]: f"{data['id']}.png"}
    elif isinstance(data, list):
        # List of items
        return {item["id"]: f"{item['id']}.png" for item in data}
    else:
        raise ValueError("Input JSON must be a dict or list of dicts.")

if __name__ == "__main__":
    # Usage: python transform_json.py input.json output.json
    if len(sys.argv) != 3:
        print("Usage: python transform_json.py input.json output.json")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    # Load input
    with open(input_path, "r") as f:
        data = json.load(f)

    # Transform
    result = transform(data)

    # Save output
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Saved transformed output to {output_path}")

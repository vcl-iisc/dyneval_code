from tifascore import tifa_score_benchmark
import json

# We recommend using mplug-large
results = tifa_score_benchmark("mplug-large", "sample/sample_question_answers.json", "sample/sample_imgs.json")

# save the results
with open("sample/sample_evaluation_result.json", "w") as f:
    json.dump(results, f, indent=4)

from tifascore import tifa_score_benchmark
import sys
import json

# We recommend using mplug-large
results = tifa_score_benchmark("mplug-large", f"/home/anirban/kanksha1/akhil/tifa/{sys.argv[1]}/tifa_v1.0_question_answers.json", f"/home/anirban/kanksha1/akhil/tifa/{sys.argv[1]}/tifa_v1.0_imgs.json")

# save the results
with open(f"/home/anirban/kanksha1/akhil/tifa/{sys.argv[1]}/tifa_v1.0_evaluation_result.json", "w") as f:
    json.dump(results, f, indent=4)

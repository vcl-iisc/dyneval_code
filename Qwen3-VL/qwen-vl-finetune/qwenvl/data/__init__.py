import os
import re


def _dataset_paths(default_annotation, default_data, annotation_env, data_env):
    return {
        "annotation_path": os.getenv(annotation_env, default_annotation),
        "data_path": os.getenv(data_env, default_data),
    }

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

## add or register the DYNEVALINSTRUCT dataset
## change the annotation_path and the data_path
## data_path is where we kept the images
## annotation_path is the json path which is having instruction and image_name within a json
DYNEVALINSTRUCT = _dataset_paths(
    "/path/to/dynevalinstruct.json",
    "/path/to/image/data",
    "DYNEVALINSTRUCT_ANNOTATION",
    "DYNEVALINSTRUCT_DATA",
)

# Stage 1 curriculum split: <|T2IA|> question generation and <|EVALUATION|> scoring samples
DYNEVALINSTRUCT_T2IA = _dataset_paths(
    "/path/to/dynevalinstruct_t2ia.json",
    "/path/to/image/data",
    "DYNEVALINSTRUCT_T2IA_ANNOTATION",
    "DYNEVALINSTRUCT_T2IA_DATA",
)

# Stage 2 curriculum split: <|IQA|> scene-graph / quality-question generation and <|EVALUATION|> scoring samples
DYNEVALINSTRUCT_IQA = _dataset_paths(
    "/path/to/dynevalinstruct_iqa.json",
    "/path/to/image/data",
    "DYNEVALINSTRUCT_IQA_ANNOTATION",
    "DYNEVALINSTRUCT_IQA_DATA",
)

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "dynevalinstruct": DYNEVALINSTRUCT,
    "dynevalinstruct_t2ia": DYNEVALINSTRUCT_T2IA,
    "dynevalinstruct_iqa": DYNEVALINSTRUCT_IQA,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)

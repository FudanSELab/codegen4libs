import pickle
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from project.util.path_util import PathUtil
from project.util.logs_util import LogsUtil

logger = LogsUtil.get_logs_util()


def filter_with_upper(version: str, upper: int):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/filter-github-code-java-libs"))["train"]
    data = []
    other_data = []
    with open(PathUtil.datasets(f"{version}/count_lib.bin"), "rb") as f:
        count_lib = pickle.load(f)
    chose_libs = count_lib.keys()
    count_lib = {_: 0 for _ in count_lib.copy().keys()}
    for row in tqdm(dataset):
        lib_size = len(row["libraries"])
        is_append = False
        if all(
            any(lib.startswith(_) for _ in ("java.", "javax.", "android.", "androidx.")) for lib in row["libraries"]
        ):
            other_data.append(row)
            continue
        if any(lib not in chose_libs for lib in row["libraries"]):
            continue
        for lib in row["libraries"]:
            if count_lib[lib] >= upper:
                continue
            # 按优先级过滤JDK&SDK
            if lib == "jdk" and lib_size > 1 or lib == "sdk" and lib_size > 2:
                continue
            if not any(lib.startswith(_) for _ in ("java.", "javax.", "android.", "androidx.")):
                count_lib[lib] += 1
            is_append = True
        if is_append:
            data.append(row)
        else:
            other_data.append(row)
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_list(data)
    dataset.save_to_disk(PathUtil.datasets(f"{version}_{upper}/filter-github-code-java-libs"))

    other_dataset = DatasetDict()
    other_dataset["train"] = Dataset.from_list(other_data)
    other_dataset.save_to_disk(PathUtil.datasets(f"{version}_{upper}/other-github-code-java-libs"))
    with open(PathUtil.datasets(f"{version}_{upper}/train-github-code-java-libs.txt"), "w") as file:
        for lib, count in count_lib.items():
            file.write(lib + ", " + str(count) + "\n")
    with open(PathUtil.datasets(f"{version}_{upper}/count_lib.bin"), "wb") as file:
        pickle.dump({lib: count for lib, count in count_lib.items()}, file)


def split_data(version: str, ration: float = 0.02, test_size: int = None):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/filter-github-code-java-libs"))["train"]
    if test_size is None:
        test_size = len(dataset) * ration
    with open(PathUtil.datasets(f"{version}/count_lib.bin"), "rb") as f:
        count_lib = pickle.load(f)
    validation_dataset, train_dataset, test_dataset = [], [], []
    lib_count_4_validation = {_: 0 for _ in count_lib.copy().keys()}
    lib_count_4_test = lib_count_4_validation.copy()
    nl_set_4_validation, nl_set_4_test = set(), set()
    for row in tqdm(dataset):
        lib_size = len(row["libraries"])
        # 按库划分数据集
        is_append_validation = False
        for lib in row["libraries"]:
            if lib not in count_lib:
                continue
            if lib_count_4_validation[lib] >= count_lib[lib] * ration:
                break
            # 按优先级过滤JDK&SDK
            if lib == "jdk" and lib_size > 1 or lib == "sdk" and lib_size > 2:
                continue
            lib_count_4_validation[lib] += 1
            is_append_validation = True
        if is_append_validation:
            validation_dataset.append(row)
            nl_set_4_validation.add(row["comment"] + row["libraries_info"])
            continue
        is_append_test = False
        for lib in row["libraries"]:
            if lib not in count_lib:
                continue
            if lib_count_4_test[lib] >= count_lib[lib] * ration:
                break
            # 按优先级过滤JDK&SDK
            if lib == "jdk" and lib_size > 1 or lib == "sdk" and lib_size > 2:
                continue
            lib_count_4_test[lib] += 1
            is_append_test = True
        if is_append_test:
            test_dataset.append(row)
            nl_set_4_test.add(row["comment"] + row["libraries_info"])
            continue
        # 同NL采集
        if (
            row["comment"] + row["libraries_info"] in nl_set_4_validation
            or row["comment"] + row["libraries_info"] in nl_set_4_test
        ):
            logger.info(row["comment"] + row["libraries_info"])
            continue
        train_dataset.append(row)
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_list(train_dataset)
    dataset["validation"] = Dataset.from_list(validation_dataset)
    dataset["test"] = Dataset.from_list(test_dataset)
    dataset.save_to_disk(PathUtil.datasets(f"{version}/github-code-java-libs"))


def slim_data(version: str):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/github-code-java-libs"))
    def chunk_examples(examples):
        return {
            "input_ids": examples["input_ids"],
            "attention_mask": examples["attention_mask"],
            "labels": examples["labels"],
        }

    dataset = dataset.map(chunk_examples, batched=True)
    dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset["train"].column_names)
    dataset.save_to_disk(PathUtil.datasets(f"{version}/slim-github-code-java-libs"))


if __name__ == "__main__":
    # with open(PathUtil.datasets(f"latest_0,800000_5000/count_lib.bin"), "rb") as file:
    #     data = pickle.load(file)
    # print(data)
    
    # filter_with_upper("latest_400000,600000", 5000)
    
    split_data("latest_0,400000_5000", ration=0.02)
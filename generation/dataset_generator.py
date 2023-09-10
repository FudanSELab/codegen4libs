import argparse
import pickle
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from project.dataset.process import DataProcessUtil
from project.util.path_util import PathUtil
from project.util.logs_util import LogsUtil

logger = LogsUtil.get_logs_util()

DATA_TYPE_TRAIN = "train"
DATA_TYPE_VALID = "validation"
DATA_TYPE_TEST = "test"


def convert_data(version: str):
    dataset = load_dataset("json", data_files=PathUtil.datasets(f"github-code-java-libs-{version}.json"))
    dataset.save_to_disk(PathUtil.datasets(f"{version}/raw-github-code-java-libs"))


def process_data(version: str, input_r: str, label_r: str):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/raw-github-code-java-libs"))
    tokenized_dataset = dataset.map(
        lambda x: DataProcessUtil.preprocess_function_with_connect(x, input_r=input_r, label_r=label_r),
        batched=True,
        load_from_cache_file=False,
    )
    tokenized_dataset.save_to_disk(PathUtil.datasets(f"{version}/processed-github-code-java-libs"))


def filter_with_token_length(version: str, max_length: int = 384):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/processed-github-code-java-libs"))
    filter_dataset = dataset.filter(
        lambda x: x["labels_token_length"] <= max_length and x["input_token_length"] <= max_length
    )
    filter_dataset.save_to_disk(PathUtil.datasets(f"{version}/filter-github-code-java-libs"))
    analyse_data(version, "train", filter_dataset, is_limit=True)


def analyse_data(version: str, typ: str, dataset, is_limit=False):
    analysis = defaultdict(int)
    jdk_sdk_analysis = defaultdict(int)
    for item in tqdm(dataset[typ]):
        libs = item["libraries"]
        for lib in libs:
            if any(lib.startswith(_) for _ in ("java", "javax", "android", "androidx")):
                jdk_sdk_analysis[lib] += 1
                continue
            analysis[lib] += 1
    analysis = sorted(analysis.items(), key=lambda x: x[1], reverse=True)
    jdk_sdk_analysis = sorted(jdk_sdk_analysis.items(), key=lambda x: x[1], reverse=True)
    analysis += jdk_sdk_analysis
    if is_limit:
        with open(PathUtil.datasets(f"{version}/count_lib.bin"), "wb") as file:
            pickle.dump({lib: count for lib, count in analysis}, file)
    with open(PathUtil.datasets(f"{version}/{typ}-github-code-java-libs.txt"), "w") as file:
        for lib, count in analysis:
            file.write(lib + ", " + str(count) + "\n")


def check_data(args, console_only: bool = False, do_analyse: bool = False):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{args.version}/{args.filename}"))
    print(dataset)
    if do_analyse:
        # 数据集分析，三方库频次统计
        analyse_data(args.version, DATA_TYPE_TRAIN, dataset)
        analyse_data(args.version, DATA_TYPE_VALID, dataset)
        analyse_data(args.version, DATA_TYPE_TEST, dataset)
    for i in range(args.check_size):
        if console_only:
            continue
        logger.info("libraries=" + dataset[DATA_TYPE_TEST][i]["comment"])
        logger.info("libraries=" + dataset[DATA_TYPE_TEST][i]["decoded_labels"])
        logger.info("libraries=" + dataset[DATA_TYPE_TEST][i]["decoded_preds"])


def split_data(version: str, ration: float = 0.02, test_size: int = None):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/filter-github-code-java-libs"))["train"]
    # dataset = dataset.train_test_split(test_size=100)["test"]
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
            if lib_count_4_validation[lib] >= count_lib[lib] * ration:
                break
            # 按优先级过滤JDK&SDK
            if lib == "jdk" and lib_size > 1 or lib == "sdk" and lib_size > 2:
                continue
            lib_count_4_validation[lib] += 1
            is_append_validation = True
        if is_append_validation:
            validation_dataset.append(row)
            continue
        is_append_test = False
        for lib in row["libraries"]:
            if lib_count_4_test[lib] >= count_lib[lib] * ration:
                break
            # 按优先级过滤JDK&SDK
            if lib == "jdk" and lib_size > 1 or lib == "sdk" and lib_size > 2:
                continue
            lib_count_4_test[lib] += 1
            is_append_test = True
        if is_append_test:
            test_dataset.append(row)
            continue
        # 同NL采集
        if row["comment"] in nl_set_4_validation or row["comment"] in nl_set_4_test:
            logger.info(row["comment"])
            continue
        train_dataset.append(row)
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_list(train_dataset)
    dataset["validation"] = Dataset.from_list(validation_dataset)
    dataset["test"] = Dataset.from_list(test_dataset)
    dataset.save_to_disk(PathUtil.datasets(f"{version}/github-code-java-libs"))


def postprocess_data(args, saved_version: str, input_r: str, label_r: str):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{args.version}/{args.filename}"))
    tokenized_dataset = dataset.map(
        lambda x: DataProcessUtil.preprocess_function_with_connect(x, input_r=input_r, label_r=label_r), batched=True
    )
    tokenized_dataset.save_to_disk(PathUtil.datasets(f"{saved_version}/github-code-java-libs"))

def add_args(parser):
    parser.add_argument(
        "--task", type=str, required=True, choices=["convert", "process", "filter", "split", "check", "postprocess"]
    )
    parser.add_argument('--version', type=str, help="The version of datasets.")
    parser.add_argument('--filename', type=str, help="The filename of dataset.")
    parser.add_argument('--check_size', type=int, help="Size of checking.")
    parser.add_argument('--input', default=None, type=str, help="Type of input.")
    parser.add_argument('--label', default=None, type=str, help="Type of label.")

    parser.add_argument('--upper', default=None, type=int, help="Max of the lib count in test split.")
    parser.add_argument('--test_size', default=4000, type=int, help="Size of the test split.")
    parser.add_argument('--saved_version', default=None, type=str, help="The version of postprocess datasets.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    versions = {
        "l0": ("nl", "code"),
        "l1": ("nl+libs", "code"),
        "l2": ("nl+libs+codeRet", "code"),
        "l3": ("nl+libs+importsGen", "code"),
        "l4": ("nl+libs+importsGen+codeRet", "code"),
        "l5": ("nl+libs", "imports"),
        "l6": ("nl+libs+importsRet", "imports")
    }

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)

    if args.task == "convert":
        # step 1
        convert_data(version=args.version)
    elif args.task == "process":
        # step 2
        process_data(version=args.version, input_r=args.input, label_r=args.label)
    elif args.task == "filter":
        # step 3
        filter_with_token_length(version=args.version)
    elif args.task == "filter" and args.upper != None:
        # step 4
        filter_with_upper(version=args.version, upper=args.upper)
    elif args.task == "split":
        # step 5
        split_data(version=args.version, test_size=args.test_size)
    elif args.task == "check":
        # step 6
        check_data(args=args)
    elif args.task == "postprocess":
        # other step
        input_label = versions[args.saved_version[:2]]
        postprocess_data(args, saved_version=args.saved_version, input_r=input_label[0], label_r=input_label[1])

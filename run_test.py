import os
import re
import json
import argparse
from datasets import DatasetDict

from project.dataset.process import DataProcessUtil
from project.metric.computer import MetricComputer
from project.util.path_util import PathUtil
from project.util.logs_util import LogsUtil

logger = LogsUtil.get_logs_util()

pattern_1 = re.compile(r"([A-Za-z]+\.[A-Za-z0-9]+\()")
pattern_2 = re.compile(r"[ |.]([a-z]+?[A-Za-z0-9]+\()")
pattern_3 = re.compile(r"[ |.]([A-Z]+?[A-Za-z0-9]*)")


def _split_2_package_class(imports_info: str):
    packages = set()
    classes = set()
    for import_info in imports_info.split(";"):
        if not import_info.strip():
            continue
        split_imports = [_ for _ in import_info.strip().split(" ")[-1].split(".") if _]
        if len(split_imports) > 0:
            i = len(split_imports) - 1
            while i > 0:
                if split_imports[i][0].isupper():
                    classes.add(split_imports[i])
                    break
                i -= 1
            package_str = ".".join(split_imports[:i]) if i > 0 else ".".join(split_imports)
            if len(package_str) > 0:
                packages.add(package_str)
    return packages, classes


def _extract_apis(code):
    class_dot_method = set([_ for _ in re.findall(pattern_1, code)])
    method_apis = set([_[_.index(".") + 1 : -1] for _ in class_dot_method])
    method_apis |= set([_[:-1] for _ in re.findall(pattern_2, code)])
    method_apis -= {"function"}
    class_apis = set([_ for _ in [_[: _.index(".")] for _ in class_dot_method] if len(_) > 0 and _[0].isupper()])
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    code = " " + code.translate(str.maketrans(dict.fromkeys("!\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~", " ")))
    class_apis |= set([_ for _ in re.findall(pattern_3, code) if not _.isupper()])
    return class_apis, method_apis


def _cal_ac(dataset, decoded_labels_typ, decoded_preds_tye):
    full_ac_cnt = 0
    half_ac_cnt = 0
    precisions = ([], [], [])
    recalls = ([], [], [])
    data_cnt = len(dataset)
    for row in dataset:
        # import
        imports_labels = [_.strip() for _ in row[decoded_labels_typ].split(";") if _.strip()]
        imports_preds = [_.strip() for _ in row[decoded_preds_tye].split(";") if _.strip()]
        # package & class
        packages_classes_labels = _split_2_package_class(row[decoded_labels_typ])
        packages_classes_preds = _split_2_package_class(row[decoded_preds_tye])

        logger.info("p=" + str(imports_preds))
        logger.info("r=" + str(imports_labels))

        # import
        true_cnt = len(set(imports_labels) & set(imports_preds))
        half_ac_cnt += 1 if true_cnt > 0 else 0
        full_ac_cnt += 1 if true_cnt == len(imports_labels) else 0
        precisions[0].append(true_cnt / len(imports_preds) if len(imports_preds) > 0 else 1)
        recalls[0].append(true_cnt / len(imports_labels))

        # package
        true_cnt = len(set(packages_classes_labels[0]) & set(packages_classes_preds[0]))
        precisions[1].append(true_cnt / len(packages_classes_preds[0]) if len(packages_classes_preds[0]) > 0 else 1)
        recalls[1].append(true_cnt / len(packages_classes_labels[0]) if len(packages_classes_labels[0]) > 0 else 1)

        # class
        true_cnt = len(set(packages_classes_labels[1]) & set(packages_classes_preds[1]))
        precisions[2].append(true_cnt / len(packages_classes_preds[1]) if len(packages_classes_preds[1]) > 0 else 1)
        recalls[2].append(true_cnt / len(packages_classes_labels[1]) if len(packages_classes_labels[1]) > 0 else 1)

    full_ac = float(full_ac_cnt / data_cnt)
    half_ac = float(half_ac_cnt / data_cnt)
    impor_precision_recall = float(sum(precisions[0])) / len(precisions[0]), float(sum(recalls[0]) / len(recalls[0]))
    packg_precision_recall = float(sum(precisions[1])) / len(precisions[1]), float(sum(recalls[1]) / len(recalls[1]))
    class_precision_recall = float(sum(precisions[2])) / len(precisions[2]), float(sum(recalls[2]) / len(recalls[2]))

    logger.info("full ac: {:.5f}".format(full_ac))
    logger.info("half ac: {:.5f}".format(half_ac))
    logger.info("impor@precision: {:.5f}\t recall: {:.5f}".format(impor_precision_recall[0], impor_precision_recall[1]))
    logger.info("packg@precision: {:.5f}\t recall: {:.5f}".format(packg_precision_recall[0], packg_precision_recall[1]))
    logger.info("class@precision: {:.5f}\t recall: {:.5f}".format(class_precision_recall[0], class_precision_recall[1]))
    print(full_ac, half_ac, impor_precision_recall[0], impor_precision_recall[1])
    print(packg_precision_recall[0], packg_precision_recall[1])
    print(class_precision_recall[0], class_precision_recall[1])


def cal_ac(version: str, decoded_labels_typ: str = "decoded_labels", decoded_preds_typ: str = "decoded_preds"):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/test-github-code-java-libs"))
    # 过滤测试代码
    # dataset = dataset.filter(
    #     lambda x: all([_ not in x["imports_info"].lower() for _ in ["junit", "assert", "test"]])
    #     and all([_ not in x["method"].lower() for _ in ["assert", "test"]])
    # )
    # _cal_ac(dataset["validation"])
    _cal_ac(dataset["test"], decoded_labels_typ, decoded_preds_typ)
    print(dataset)


def _cal_metrics(dataset, decoded_labels_typ, decoded_preds_typ):
    MetricComputer.compute_decoded_metrics((dataset[decoded_preds_typ], dataset[decoded_labels_typ]))


def cal_metrics(version: str, decoded_labels_typ: str = "decoded_labels", decoded_preds_typ: str = "decoded_preds"):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/test-github-code-java-libs"))
    # 过滤测试代码
    # dataset = dataset.filter(
    #     lambda x: all([_ not in x["imports_info"].lower() for _ in ["junit", "assert", "test"]])
    #     and all([_ not in x["method"].lower() for _ in ["assert", "test"]])
    # )
    # dataset = dataset.filter(lambda x: x["libraries"] != ["jdk"] and x["libraries"] != ["sdk"])
    print(dataset)
    _cal_metrics(dataset["test"], decoded_labels_typ, decoded_preds_typ)


def _cal_precision_recall(preds, labels, full_ac_cnt, half_ac_cnt, precisions, recalls):
    true_cnt = len(set(labels) & set(preds))
    full_ac_cnt += 1 if true_cnt == len(labels) else 0
    half_ac_cnt += 1 if true_cnt > 0 else 0
    precisions.append(true_cnt / len(preds) if len(preds) > 0 else 1)
    recalls.append(true_cnt / len(labels) if len(labels) > 0 else 1)
    return full_ac_cnt, half_ac_cnt


def _cal_mismatch(dataset, decoded_typ):
    # 0: class, 1: method, 2: class_by_main_lib
    precisions = ([], [], [])
    recalls = ([], [], [])
    full_ac_cnt = [0, 0, 0]
    half_ac_cnt = [0, 0, 0]
    data_cnt = len(dataset)
    for row in dataset:
        # class & method
        class_method_p = _extract_apis(row[decoded_typ])
        class_method_r = _extract_apis(row["decoded_labels"])
        full_ac_cnt[0], half_ac_cnt[0] = _cal_precision_recall(
            class_method_p[0], class_method_r[0], full_ac_cnt[0], half_ac_cnt[0], precisions[0], recalls[0]
        )
        full_ac_cnt[1], half_ac_cnt[1] = _cal_precision_recall(
            class_method_p[1], class_method_r[1], full_ac_cnt[1], half_ac_cnt[1], precisions[1], recalls[1]
        )

        libs = []
        for lib in row["libraries"]:
            if lib == 'jdk':
                libs.append("java")
            elif lib == "sdk":
                libs.append("android")
            else:
                libs.append(lib)
        bad_classes = _split_2_package_class(
            ";".join([import_ for import_ in row["imports"] if all([_ not in import_ for _ in libs])])
        )[1]
        main_lib_related_class = _split_2_package_class(
            ";".join([import_ for import_ in row["imports"] if any([_ in import_ for _ in libs])])
        )[1]
        generated_class = _extract_apis(row[decoded_typ])[0] - set(bad_classes)
        full_ac_cnt[2], half_ac_cnt[2] = _cal_precision_recall(
            generated_class, main_lib_related_class, full_ac_cnt[2], half_ac_cnt[2], precisions[2], recalls[2]
        )

    full_ac = float(full_ac_cnt[0] / data_cnt), float(full_ac_cnt[1] / data_cnt), float(full_ac_cnt[2] / data_cnt)
    half_ac = float(half_ac_cnt[0] / data_cnt), float(half_ac_cnt[1] / data_cnt), float(half_ac_cnt[2] / data_cnt)

    class_pandr = float(sum(precisions[0])) / len(precisions[0]), float(sum(recalls[0]) / len(recalls[0]))
    class_f1 = 2 * (class_pandr[0] * class_pandr[1]) / (class_pandr[0] + class_pandr[1])

    metho_pandr = float(sum(precisions[1])) / len(precisions[1]), float(sum(recalls[1]) / len(recalls[1]))
    metho_f1 = 2 * (metho_pandr[0] * metho_pandr[1]) / (metho_pandr[0] + metho_pandr[1])

    by_main_lib_pandr = float(sum(precisions[2])) / len(precisions[2]), float(sum(recalls[2]) / len(recalls[2]))
    by_main_lib_f1 = 2 * (by_main_lib_pandr[0] * by_main_lib_pandr[1]) / (by_main_lib_pandr[0] + by_main_lib_pandr[1])

    logger.info("class@full ac: {:.5f}".format(full_ac[0]))
    logger.info("class@half ac: {:.5f}".format(half_ac[0]))
    logger.info(
        "class@precision: {:.5f}\t recall: {:.5f}\t f1: {:.5f}".format(class_pandr[0], class_pandr[1], class_f1)
    )
    logger.info("metho@full ac: {:.5f}".format(full_ac[1]))
    logger.info("metho@half ac: {:.5f}".format(half_ac[1]))
    logger.info(
        "metho@precision: {:.5f}\t recall: {:.5f}\t f1: {:.5f}".format(metho_pandr[0], metho_pandr[1], metho_f1)
    )
    logger.info("classFilter@full ac: {:.5f}".format(full_ac[2]))
    logger.info("classFilter@half ac: {:.5f}".format(half_ac[2]))
    logger.info(
        "classFilter@precision: {:.5f}\t recall: {:.5f}\t f1: {:.5f}".format(
            by_main_lib_pandr[0], by_main_lib_pandr[1], by_main_lib_f1
        )
    )
    print(full_ac[0], half_ac[0], class_pandr[0], class_pandr[1], class_f1)
    print(full_ac[1], half_ac[1], metho_pandr[0], metho_pandr[1], metho_f1)
    print(full_ac[2], half_ac[2], by_main_lib_pandr[0], by_main_lib_pandr[1], by_main_lib_f1)


def cal_mismatch(version: str, decoded_typ: str = "decoded_preds"):
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/test-github-code-java-libs"))
    # 过滤测试代码
    # dataset = dataset.filter(
    #     lambda x: all([_ not in x["imports_info"].lower() for _ in ["junit", "assert", "test"]])
    #     and all([_ not in x["method"].lower() for _ in ["assert", "test"]])
    # )
    # _cal_metrics(dataset["validation"])
    _cal_mismatch(dataset["test"], decoded_typ)
    print(dataset)


def eval_test(version: str, datasets_version: str, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not datasets_version:
        datasets_version = version
    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{datasets_version}/github-code-java-libs"))
    test_validation_dataset = DatasetDict()
    test_validation_dataset["test"] = dataset["test"]
    test_validation_dataset["validation"] = dataset["validation"]
    test_validation_dataset = test_validation_dataset.map(
        lambda x: DataProcessUtil.postprocess_function_with_generate(x, version), batched=True, batch_size=32
    )
    test_validation_dataset.save_to_disk(PathUtil.datasets(f"{datasets_version}/test-github-code-java-libs"))
    test_dataset = test_validation_dataset["test"]
    MetricComputer.compute_decoded_metrics((test_dataset["decoded_preds"], test_dataset["decoded_labels"]))


def add_args(parser):
    parser.add_argument("--version", type=str, required=True, help="The version of model.")

    parser.add_argument("--gpu", default=0)
    parser.add_argument(
        "--datasets_version", default=None, type=str, help="The version of datasets. Same as version if None."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    version_ = "l1"
    # code
    cal_metrics(version_)
    # imports
    cal_metrics(version_, "imports_info", "retrieved_imports_info")
    cal_metrics(version_, "imports_info", "generated_imports_info")
    cal_metrics(version_, "imports_info", "cleaner_generated_imports_info")
    cal_metrics(version_, "imports_info", "union_gen_rei_imports_info")
    cal_metrics(version_, "imports_info", "intersection_gen_rei_imports_info")


    # imports
    cal_ac(version_)
    cal_ac(version_, "imports_info", "retrieved_imports_info")
    cal_ac(version_, "imports_info", "generated_imports_info")
    cal_ac(version_, "imports_info", "cleaner_generated_imports_info")
    cal_ac(version_, "imports_info", "union_gen_rei_imports_info")
    cal_ac(version_, "imports_info", "intersection_gen_rei_imports_info")

    # code
    cal_mismatch(version_)
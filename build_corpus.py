import json
import jsonlines
import pickle
import string
from collections import defaultdict
import random

import re
from tqdm import tqdm

from script.extract_method import ProjectMethodExtractor

from sckg.util.path_util import PathUtil
from sckg.util.log_util import LogUtil

logger = LogUtil.get_log_util()

multi_comment_pattern = re.compile(r"/\*(.+?)\*/", re.DOTALL)
single_comment_pattern = re.compile(r" *?//.+?\r?\n", re.DOTALL)
at_annotation_pattern = re.compile(r"@[a-zA-Z]+?\r?\n", re.DOTALL)
package_pattern = re.compile(r"package (.+?);\r?\n", re.DOTALL)
library_pattern = re.compile(r"import\s.*?([a-zA-Z0-9_.*]+?);\r?\n", re.DOTALL)

doc_pattern_1 = re.compile(r"\{@[a-zA-Z]+? (.+?)\}")
doc_pattern_2 = re.compile(r"\{@.+?\}")
doc_link_pattern = re.compile(r"\{@link (.+?)\}")
tag_pattern = re.compile(r"<.+?>")
href_pattern = re.compile(r'((https?):((//)|(\\\\))+([\w0-9#@%/;$~_?\+-=\\\.&](#!)?)*)')

api_pattern_1 = re.compile(r"([A-Za-z]+\.[A-Za-z0-9]+\()")
api_pattern_2 = re.compile(r"[ |.]([a-z]+?[A-Za-z0-9]+\()")
api_pattern_3 = re.compile(r"[ |.]([A-Z]+?[A-Za-z0-9]*)")


def extract_comment(text):
    """
    :param text: /**...*/中的注释
    :return: 清洗后的NL
    """
    if not text:
        return ""
    find_comments = multi_comment_pattern.findall(text)
    if not find_comments:
        return ""
    find_comments = re.findall(r"\*(.*?)\n", find_comments.copy()[-1])
    comments = []
    for comment in find_comments:
        comment = comment.strip().strip("/*")
        if not comment:
            continue
        if comment.startswith("@"):
            # eg. "@param"; "@return";
            break
        # eg: "NOTE:";
        if "* @" in comment:
            comment = comment.split("* @")[0]
        if re.findall(re.compile(r"[A-Z]+?:"), comment):
            continue
        comments.append(comment)
    comment = " ".join(comments).replace("\n", " ")
    # eg. "{@link XXX}"; "{@inheritDoc}";
    comment = re.sub(
        doc_pattern_2, "", re.sub(doc_pattern_1, "\g<1>", re.sub(doc_link_pattern, "<code>\g<1></code>", comment))
    )
    # <CODE></CODE> => <code></code>
    comment = comment.replace("<CODE>", "<code>").replace("</CODE>", "</code>")
    # <pre></pre> => <code></code>
    comment = comment.replace("<pre>", "<code>").replace("</pre>", "</code>")
    tags = re.findall(tag_pattern, comment)
    for tag in tags:
        if tag in ("<code>", "</code>"):
            continue
        comment = comment.replace(tag, "")
    # eg. "http://"; "https://"
    comment = re.sub(href_pattern, "HREF", comment)
    # 句末标点
    if comment and comment[-1] in ",.;!?~":
        comment = comment[:-1]
    comment = " ".join(comment.split()).strip()
    if any([_ in comment for _ in ("See #", "Bug#", "BUG#", "issue #", "Issue #")]):
        return ""
    return comment.strip()


def longest_comment_prefix(str1, str2):
    length, index = min(len(str1), len(str2)), 0
    while index < length and str1[index] == str2[index]:
        index += 1
    return str1[:index]


def filter_self_imports(package, imports):
    filter_libs = []
    for import_item in imports:
        if longest_comment_prefix(package, import_item):
            continue
        filter_libs.append(import_item)
    return filter_libs


def extract_import_info(code, extra: dict = None):
    package = package_pattern.search(code)
    package = package.group(1) if package else ""
    find_libs = library_pattern.findall(code)
    if not find_libs:
        return []
    find_libs = [_.split(".") for _ in find_libs.copy()]
    find_libs = filter_self_imports(package, find_libs)
    api_2_imports = dict()
    # 按class:imports记录所有improst信息
    for _ in find_libs:
        api_2_imports.setdefault(_[-1].strip(), []).append(".".join(_))
    methods, names, docs = ProjectMethodExtractor.extract_methods_with_doc_from_class(code)
    for method, name, doc in zip(methods, names, docs):
        # 优先过滤没有注释的代码
        comment = extract_comment(doc)
        if not comment or not is_en(comment):
            continue
        method = re.sub(multi_comment_pattern, "", method).strip(" ").strip("\n")
        sub_method = re.sub(re.compile(r"[\s|\r|\n|\t]+"), " ", method).strip()
        info = {"method": method, "clean_method": sub_method, "doc": doc, "comment": comment, "method_name": name}
        if extra:
            info["extra"] = extra
        info["api_2_imports"] = api_2_imports
        yield info


def preprocess(filename):
    with open(PathUtil.orin_datasets(f"{filename}.json"), "r") as file:
        data = json.load(file)
    process_data = []
    for item in tqdm(data):
        code = item["code"]
        extra = {"repo_name": item["repo_name"], "path": item["path"], "license": item["license"], "size": item["size"]}
        try:
            batch_data = [_ for _ in extract_import_info(code, extra)]
        except Exception as e:
            logger.info(e, code)
            continue
        if not batch_data:
            continue
        process_data.extend(batch_data)
    with open(PathUtil.datasets(f"preprocess_{filename}.json"), "w") as file:
        json.dump(process_data, file)


def extract_api(code):
    class_dot_method = set([_ for _ in re.findall(api_pattern_1, code)])
    method_apis = set([_[_.index(".") + 1 : -1] for _ in class_dot_method])
    method_apis |= set([_[:-1] for _ in re.findall(api_pattern_2, code)])
    method_apis -= {"function"}
    class_apis = set([_ for _ in [_[: _.index(".")] for _ in class_dot_method] if len(_) > 0 and _[0].isupper()])
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    code = " " + code.translate(str.maketrans(dict.fromkeys("!\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~", " ")))
    class_apis |= set([_ for _ in re.findall(api_pattern_3, code) if not _.isupper()])
    return class_apis, method_apis


def format_import_info(split_import_info):
    i = 0
    while i < len(split_import_info):
        if split_import_info[i][0].isupper():
            break
        i += 1
    return split_import_info[: i + 1]


def cluster_libs(filename, group_prefixes):
    with open(PathUtil.datasets(f"{filename}.json"), "r") as file:
        data = json.load(file)
    cluster_data = []
    for item in tqdm(data):
        # clean-method 二次处理
        method = item["method"]
        # 单行注释及注解清理
        method = re.sub(multi_comment_pattern, "", method)
        clean_method = re.sub(
            re.compile(r"[\s|\r|\n|\t]+"),
            " ",
            re.sub(single_comment_pattern, "", re.sub(at_annotation_pattern, "", method + "\n")),
        ).strip()
        # 过滤类级别
        if re.search(re.compile(r"(public|abstract) class"), clean_method):
            item["clean_method"] = ""
            continue
        # 过滤抽象方法
        if re.search(re.compile(r"abstract (.*?) (%s)" % item["method_name"]), clean_method):
            item["clean_method"] = ""
            continue
        # 统一长字符串
        ss = re.findall(re.compile(r'".+?"'), clean_method)
        for s in ss:
            if len(s) > 3 and len(s.split()) > 1 or len(s) > 7:
                clean_method = clean_method.replace(s, "STR")
        # 修饰符及方法名处理
        modifier_pattern = re.compile(r"(public|protected|private|abstract) (.*?) (%s)" % item["method_name"])
        clean_method = re.sub(modifier_pattern, "\g<2> function", clean_method)
        if clean_method and clean_method[0] in r"""!"#$%&'()*+,-./:;<=>?[\]^_`{|}~""":
            clean_method = ""
        item["clean_method"] = clean_method

        # 清理imports异常
        find_imports = [_ for imports in item["api_2_imports"].values() for _ in imports]
        api_2_imports = dict()
        for find_import in find_imports:
            find_import = re.sub(r"\.+", ".", find_import)
            find_import = find_import.strip().strip(".").split(".")
            if len(find_import) < 2:
                continue
            api_2_imports.setdefault(find_import[-1].strip(), []).append(".".join(find_import))
        item["api_2_imports"] = api_2_imports

        # 抽取imports-info
        orig_import_infos = []
        class_apis, method_apis = extract_api(clean_method)
        # 过滤代码中未提及class&method的imports信息
        for api, imports in api_2_imports.items():
            if api in class_apis | method_apis:
                orig_import_infos.extend(imports)
        if not orig_import_infos:
            continue
        libs = set()
        imports = set()
        cluster_import_dic = {}
        for import_info in orig_import_infos:
            split_import_info = import_info.split(".")
            new_import_info = format_import_info(split_import_info)
            idx = 0
            while idx < len(split_import_info) and split_import_info[idx].islower():
                idx += 1
            split_import_info = split_import_info[:idx]
            if len(split_import_info) < 2:
                continue
            # 前缀过滤
            group_prefix = split_import_info[0]
            if group_prefix not in group_prefixes:
                continue
            # add imports-info
            imports.add(".".join(new_import_info))
            # add libs
            cluster_import_dic.setdefault(".".join(split_import_info), []).append(new_import_info)
            # split_import_info = ["jdk"] if group_prefix in ("java", "javax") else split_import_info
            # split_import_info = ["sdk"] if group_prefix in ("android", "androidx") else split_import_info
            split_import_info = (
                split_import_info[:2]
                if group_prefix in ("java", "javax", "android", "androidx")
                else split_import_info[:3]
            )
            libs.add(".".join(split_import_info))
        if not imports:
            continue
        libs = sorted(libs)
        imports = sorted(imports)
        # 同库缩简
        cluster_imports = [simple_name + ".*" for simple_name, _ in cluster_import_dic.items()]
        item["imports"] = imports
        item["imports_info"] = "; ".join(["import " + _ for _ in imports]) + ";"
        item["cluster_imports_info"] = "; ".join(sorted(set(["import " + _ for _ in cluster_imports]))) + ";"
        item["libraries"] = libs
        item["libraries_info"] = "; ".join(libs) + ";"
        cluster_data.append(item)
    with open(PathUtil.datasets(f"cluster_{filename}.json"), "w") as file:
        json.dump(cluster_data, file)


def load_data(prefix: str, index):
    for _, filename in PathUtil.walk_file_path(PathUtil.datasets_dir()):
        if not filename.startswith(prefix):
            continue
        filename = filename[:-5]
        i_ = filename[36:]
        if int(i_) > index:
            continue
        with open(PathUtil.datasets(f"{filename}.json"), "r") as file:
            data = json.load(file)
        logger.info("row 0=" + str(data[0]))
        for item in tqdm(data):
            if not item['libraries']:
                continue
            yield item


def combine_data(index: int = 100):
    idx = 0
    combined_data = []
    for item in load_data("cluster", index):
        # 过滤*
        if "*" in item["api_2_imports"]:
            continue
        if not item["comment"]:
            continue
        if not item["clean_method"]:
            continue
        if not item["imports_info"]:
            continue
        nl_len = len(item["comment"].split())
        if nl_len < 2 or nl_len > 48:
            continue
        item['id'] = idx
        idx += 1
        combined_data.append(item)
    with open(PathUtil.datasets(f"github-code-java-libs.json"), "w") as file:
        json.dump(combined_data, file)


def analyse_data(filename: str, lower: int = 0, is_main: bool = False, is_save: bool = True):
    analysis = defaultdict(int)
    jdk_sdk_analysis = defaultdict(int)
    with open(PathUtil.datasets(f"{filename}.json"), "r") as file:
        data = json.load(file)
    for item in tqdm(data):
        libs = item["libraries"]
        for lib in libs:
            if any(lib.startswith(_) for _ in ("java.", "javax.", "android.", "androidx.")):
                jdk_sdk_analysis[lib] += 1
                continue
            analysis[lib] += 1
    analysis = sorted(analysis.items(), key=lambda x: x[1], reverse=True)
    jdk_sdk_analysis = sorted(jdk_sdk_analysis.items(), key=lambda x: x[1], reverse=True)
    limit_lib = list(filter(lambda x: x[1] >= lower, analysis))
    limit_lib += jdk_sdk_analysis
    if not is_save:
        for item in analysis:
            print(item)
        print('data:', sum([c for _, c in analysis]))
        print('libs:', len(limit_lib))
        return
    with open(PathUtil.datasets(f"count_lib_{filename}.txt"), "w") as file:
        for lib, count in limit_lib:
            file.write(lib + ", " + str(count) + "\n")
        file.write("\n")
        file.write("data: " + str(sum([c for _, c in analysis])) + "\n")
        file.write("libs: " + str(len(limit_lib)) + "\n")
    if is_main:
        return
    with open(PathUtil.datasets("lib_zero.bin"), "wb") as file:
        pickle.dump({lib: 0 for lib, _ in limit_lib}, file)


def filter_with_upper(upper: int):
    filter_data = []
    with open(PathUtil.datasets("lib_zero.bin"), "rb") as f:
        lib_count_dic = pickle.load(f)
    chose_libs = lib_count_dic.keys()
    # print(chose_libs)
    with open(PathUtil.datasets(f"latest-slim-github-code-java-libs-clean.json"), "r") as file:
        data = json.load(file)
    for item in tqdm(data):
        lib_size = len(item["libraries"])
        is_append = False
        # libs = []
        if any(lib not in chose_libs for lib in item["libraries"]):
            continue
        for lib in item["libraries"]:
            if lib_count_dic[lib] >= upper:
                continue
            # 按优先级过滤JDK&SDK
            if lib == "jdk" and lib_size > 1 or lib == "sdk" and lib_size > 2:
                continue
            # libs.append(lib)
            lib_count_dic[lib] += 1
            is_append = True
        if is_append:
            # item["libraries"] = libs
            filter_data.append(item)
    # with open(PathUtil.datasets(f"count_lib-{upper}.bin"), "wb") as file:
    #     pickle.dump(lib_count_dic, file)
    random.shuffle(filter_data)
    with open(PathUtil.datasets(f"latest-slim-github-code-java-libs-{upper}.json"), "w") as file:
        json.dump(filter_data, file)


def slimming(upper: int = None):
    with open(PathUtil.datasets(f"github-code-java-libs.json"), "r") as file:
        data = json.load(file)
    for item in tqdm(data):
        del item["api_2_imports"]
    random.shuffle(data)
    logger.info("row 0=" + str(data[0]))
    with open(PathUtil.datasets(f"slim-github-code-java-libs.json"), "w") as file:
        json.dump(data, file)


def analyse(version: str):
    analyse_data(f"train-github-code-java-libs-{version}", is_main=True)
    analyse_data(f"validation-github-code-java-libs-{version}", is_main=True)
    analyse_data(f"test-github-code-java-libs-{version}", is_main=True)


def dump_dataset(chunksize: int = 100000, dump_count=1000000):
    data = []
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Java"])
    count = 0
    for d in tqdm(iter(ds.take(dump_count))):
        count += 1
        data.append(d)
        if count % chunksize == 0:
            with open(PathUtil.orin_datasets(f"github-code-java-{int(count / chunksize)}.json"), "w") as file:
                json.dump(data, file)
            data = []
    if data:
        with open(PathUtil.orin_datasets(f"github-code-java-{int(count / chunksize) + 1}.json"), "w") as file:
            json.dump(data, file)


def is_en(text):
    text = text.translate(str.maketrans('', '', string.punctuation + string.whitespace + string.digits))
    for _ in text:
        if not '\u0041' <= _ <= '\u005A' and not '\u0061' <= _ <= '\u007A':
            return False
    return True


if __name__ == '__main__':
    dump_dataset()

    """
    for _, filename_ in PathUtil.walk_file_path(PathUtil.orig_datasets_dir()):
        filename_ = filename_[:-5]
        i = filename_[17:]
        if int(i) <= ?_:
            continue
        print(filename_)
        preprocess(filename_)

    with open("lib_group.bin", "rb") as f:
        group_prefixes_ = pickle.load(f)
    for _, filename_ in PathUtil.walk_file_path(PathUtil.datasets_dir()):
        cluster_libs(filename_[:-5], group_prefixes_)

    combine_data()
    """

    # for _, filename_ in PathUtil.walk_file_path(PathUtil.orig_datasets_dir()):
    #     filename_ = filename_[:-5]
    #     i = filename_[17:]
    #     if int(i) <= 0:
    #         continue
    #     print(filename_)
    #     preprocess(filename_)

    # with open(PathUtil.datasets("lib_group.bin"), "rb") as f:
    #     group_prefixes_ = pickle.load(f)
    # for _, filename_ in PathUtil.walk_file_path(PathUtil.datasets_dir()):
    #     if not filename_.startswith("preprocess"):
    #         continue
    #     print(filename_)
    #     cluster_libs(filename_[:-5], group_prefixes_)

    # print(format_import_info("java.io.IOException".split(".")))

    # combine_data()
    # slimming()
    
    # analyse_data("latest-slim-github-code-java-libs-clean", 615)
    # with open(PathUtil.datasets("lib_zero.bin"), "rb") as file:
    #     data = pickle.load(file)
    # print(data)

    # filter_with_upper(20000)
    # analyse_data("latest-slim-github-code-java-libs-20000", is_main=True)

    # with open(PathUtil.datasets(f"latest-slim-github-code-java-libs-20000.json"), "r") as file:
    #     data = json.load(file)
    # filter_data = data[0:600000]
    # with open(PathUtil.datasets(f"latest-slim-github-code-java-libs-0,600000.json"), "w") as file:
    #     json.dump(filter_data, file)
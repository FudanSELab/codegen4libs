import jsonlines
import pickle
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from datasets import DatasetDict

from project.util.path_util import PathUtil
from project.util.logs_util import LogsUtil

logger = LogsUtil.get_logs_util()
STEP_SIZE = 50


class ES:
    def __init__(self, host: str = "localhost", port: int = 9200):
        self.client = Elasticsearch(f"http://{host}:{port}", request_timeout=500, max_retries=10, retry_on_timeout=True)

    def _create_index(self, index):
        if not self.client.indices.exists(index=index):
            try:
                self.client.indices.create(index=index)
            except Exception as e:
                logger.error(e.__class__.__name__, e)
        # else: self.client.indices.delete(index=index)
        return index

    def generate_doc(self, index, version: str, batch: int = 5000):
        index = self._create_index(index)
        dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version}/github-code-java-libs"))["train"]
        data_size = len(dataset)
        logger.info("data size: " + str(data_size))
        for i in range(0, data_size, batch):
            action = [
                {
                    "_index": index,
                    "_op_type": "create",
                    "_source": {
                        "nl": dataset[idx]["comment"],
                        "libs": dataset[idx]["libraries_info"],
                        "imports": dataset[idx]["imports_info"],
                        "id": dataset[idx]["id"],
                    },
                }
                for idx in tqdm(range(i, i + batch if i + batch < data_size else data_size))
            ]
            helpers.bulk(self.client, action)

    def generate_doc_itr(self, index, data: list):
        batch = len(data)
        action = [
            {
                "_index": index,
                "_op_type": "create",
                "_source": {
                    "nl": data[idx]["comment"],
                    "libs": data[idx]["libraries_info"],
                    "imports": data[idx]["imports_info"],
                    "id": data[idx]["id"],
                },
            }
            for idx in tqdm(range(0, batch))
        ]
        helpers.bulk(self.client, action)

    def _query(self, index, nl: str, from_: int = 0, size: int = 10):
        try:
            res = self.client.search(index=index, query={"match": {"nl": nl}}, from_=from_, size=size)
            if not res:
                return None
            return res["hits"]["hits"]
        except Exception as e:
            logger.error(repr(e))

    def query_imports_info_filter_n(self, index, nl: str, libs: list, n: int = 1):
        from_ = 0
        is_find = 0
        imports_infos = []
        while is_find < n and from_ < 1000:
            libs2imports = [
                (_["_source"]["libs"], _["_source"]["imports"], _["_source"]["nl"], _["_source"]["id"])
                for _ in self._query(index, nl, from_=from_, size=STEP_SIZE)
            ]
            if not libs2imports:
                logger.error("nl: " + nl)
                return []
            from_ += STEP_SIZE
            for item in libs2imports:
                if all(_ in item[0] for _ in libs) and nl != item[2]:
                    imports_infos.append(item[1])
                    is_find += 1
                if is_find >= n:
                    break
        if not imports_infos:
            return ""
        return imports_infos[:n]


if __name__ == "__main__":
    es = ES()
    # version_ = "latest_combine_0_600000_5000"
    index_ = "github-code-java-libs-2916582"

    # es.client.indices.delete(index=version_)
    # es.generate_doc(version_, version=version_)

    dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version_}/test-github-code-java-libs"))
    retrieved_dataset = DatasetDict()
    retrieved_dataset["train"] = dataset["train"]
    retrieved_dataset["test"] = dataset["test"]    
    retrieved_dataset["validation"] = dataset["validation"]

    def retrieve_imports_info(examples):
        examples["importsRet"] = [
            es.query_imports_info_filter_n(index=index_, nl=nl, libs=libs, n=1)
            for nl, libs in zip(examples["comment"], examples["libraries"])
        ]
        return examples
    
    retrieved_dataset = retrieved_dataset.map(retrieve_imports_info, batched=True, batch_size=10)
    retrieved_dataset.save_to_disk(PathUtil.datasets(f"{version_}/ret-github-code-java-libs"))

    # retrieved_dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"{version_}/retrieved_github-code-java-libs"))
    # for row in retrieved_dataset["test"]:
    #     logger.info("p=" + str(row["retrieved_imports_info"]))
    #     logger.info("r=" + str(row["imports_info"]))

    # batch = 5000
    # dataset = DatasetDict.load_from_disk(PathUtil.datasets(f"top_400000_5000/github-code-java-libs"))["train"]
    # data_size = len(dataset)
    # for i in range(260000, data_size, batch):
    #     for idx in tqdm(range(i, i + batch if i + batch < data_size else data_size)):
    #         print(dataset[idx])

    # 全集构建
    # index_ = "github-code-java-libs-2916582"
    # batch_ = 5000
    # with open(PathUtil.datasets("github-code-java-libs-2916582.jsonl"), "r+", encoding="utf8") as file:
    #     cnt = 0
    #     batch_data = []
    #     for item in jsonlines.Reader(file):
    #         batch_data.append(item)
    #         cnt += 1
    #         if (cnt >= batch_):
    #             es.generate_doc_itr(index_, batch_data)
    #             cnt = 0
    #             batch_data = []
    #     es.generate_doc_itr(index_, batch_data)

    # id2clean_method = dict()
    # with open(PathUtil.datasets("github-code-java-libs-2916582.jsonl"), "r+", encoding="utf8") as file:
    #     for item in tqdm(jsonlines.Reader(file)):
    #         id2clean_method[item["id"]] = item["clean_method"]
    # with open(PathUtil.datasets("id2clean_method.bin"), "wb") as file:
    #     pickle.dump(id2clean_method, file)
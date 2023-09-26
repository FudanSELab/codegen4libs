# CodeGen4Libs
This repo is for the ASE2023 paper titled ["CodeGen4Libs: A Two-stage Approach for Library-oriented Code Generation"](https://mingwei-liu.github.io/publication/2023-08-18-ase-CodeGen4Libs).

### Updates
***
- 2023-09-10: Initial Benchmark Release
### TODO
***
- Huggingface support
- Model Implementations
  
### Benchmark Format
***
Benchmark has been meticulously structured and saved in the DatasetDict format, accessible at [Dataset and Models of CodeGen4Libs](https://zenodo.org/record/7920906). The specific data fields for each tuple are delineated as follows:

- id: the unique identifier for each tuple.
- method: the original method-level code for each tuple.
- clean_method: the ground-truth method-level code for each task.
- doc: the document of method-level code for each tuple.
- comment: the natural language description for each tuple.
- method_name: the name of the method.
- extra: extra information on the code repository to which the method level code belongs.
    - license: the license of code repository.
    - path: the path of code repository.
    - repo_name: the name of code repository.
    - size: the size of code repository.
- imports_info: the import statements for each tuple.
- libraries_info: the libraries info for each tuple.

- input_str: the design of model input.
- input_ids: the ids of tokenized input.
- tokenized_input_str: the tokenized input.
- input_token_length: the length of the tokenized input.
- labels: the ids of tokenized output.
- tokenized_labels_str: the tokenized output.
- labels_token_length: the length of the the tokenized output.

- retrieved_imports_info: the retrieved import statements for each tuple.
- generated_imports_info: the generated import statements for each tuple.
- union_gen_ret_imports_info: the union of retrieved import statements and the generated import statements.
- intersection_gen_ret_imports_info: the intersection of retrieved import statements and the generated import statements.
- similar_code: the retrieved method-level code for each tuple.

### Models Download
***
[NL+Libs+Imports(Ret)->Imports](https://zenodo.org/record/7920906)

[NL+Libs->Imports](https://zenodo.org/record/7920906)

[NL+Libs->Code](https://zenodo.org/record/7920906)

[NL+Libs+Imports(Gen)->Code](https://zenodo.org/record/7920906)

[NL+Libs+Code(Ret)->Code](https://zenodo.org/record/7920906)

[NL+Libs+Imports(Gen)+Code(Ret)->Code](https://zenodo.org/record/7920906)

### Usage
***
1. Environment Setup
``` Python
from transformers import RobertaTokenizer, T5ForConditionalGeneration
```

2. Load Model
``` Python
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
# add <code>, </code> as special tokens
tokenizer.add_special_tokens(
    {"additional_special_tokens": tokenizer.special_tokens_map["additional_special_tokens"] + ["<code>", "</code>"]}
)
# load model
model_name = "codegen4lib_base"
model_dir = PathUtil.finetune_model(f"{version}/best_{model_name}")
model = T5ForConditionalGeneration.from_pretrained(model_dir)
```

3. Genetrate Example
``` Python
input_str = "Gets the detailed information for a given agent pool"
input_ids = tokenizer(input_str, return_tensors="pt").input_ids
input_ids = torch.as_tensor(input_ids).to("cuda")

outputs = model.generate(input_ids, max_length=512)
print("output_str: ", tokenizer.decode(outputs[0], skip_special_tokens=True))
```
### Citation

@inproceedings{ase2023codegen4libs,
  author       = {Mingwei Liu and Tianyong Yang and Yiling Lou and Xueying Du and Ying Wang and and Xin Peng},
  title        = {{CodeGen4Libs}: A Two-stage Approach for Library-oriented Code Generation},
  booktitle    = {38th {IEEE/ACM} International Conference on Automated Software Engineering,
                  {ASE} 2023, Kirchberg, Luxembourg, September 11-15, 2023},
  pages        = {0--0},
  publisher    = {{IEEE}},
  year         = {2023},
}

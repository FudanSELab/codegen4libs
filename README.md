# CodeGen4Libs

### Benchmark Format
Benchmark has been meticulously structured and saved in the DatasetDict format, accessible at [Dataset and Models of CodeGen4Libs](https://zenodo.org/record/7920906#.ZFyPm-xByDV). The specific data fields for each tuple are delineated as follows:

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
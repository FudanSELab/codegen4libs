# CodeGen4Libs

### Benchmark Format
Benchmark has been meticulously structured and saved in the DatasetDict format, accessible at [Dataset and Models of CodeGen4Libs](https://zenodo.org/record/7920906#.ZFyPm-xByDV). The specific data fields for each task are delineated as follows:

- id
- method
- clean_method
- doc
- comment
- method_name
- extra
    - license
    - path
    - repo_name
    - size
- imports_info
- libraries_info

- input_str
- attention_mask
- input_ids
- tokenized_input_str
- input_token_length
- labels
- tokenized_labels_str
- labels_token_length

- retrieved_imports_info
- generated_imports_info
- union_gen_ret_imports_info
- intersection_gen_ret_imports_info
- similar_code

- decoded_labels
- predictions
- decoded_preds
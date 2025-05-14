import os

import shutil
import ast
import re
import pandas as pd

prompt_pat = r'^\s*(>>> ?|\.\.\. ?|\\$|In\s*\[\d*\]:\s*)'
slash_pat  = r'^\s*/.*'

# 두 패턴을 하나로 OR 결합
prompt_re = re.compile(f'{prompt_pat}|{slash_pat}', re.MULTILINE)

def strip_prompts(code: str) -> str:
    lines = code.split('\n')
    lines = [prompt_re.sub('', line) for line in lines]
    if lines:
        lines[0] = lines[0].lstrip()
    cleaned = '\n'.join(lines)
    return cleaned

import tokenize
from io import StringIO
def strip_prompts(code: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        lines = code.split('\n')
        lines = [re.sub(prompt_re, '', line) for line in lines]
        if lines:
            lines[0] = lines[0].lstrip()
        return '\n'.join(lines)     

    # 줄 수를 동적으로 확장하며 저장
    line_map = {}
    for tok_type, tok_str, (start_row, _), (_, _), _ in tokens:
        if tok_type == tokenize.STRING:
            line_map[start_row] = line_map.get(start_row, '') + tok_str
        else:
            cleaned = prompt_re.sub('', tok_str)
            line_map[start_row] = line_map.get(start_row, '') + cleaned

    max_line = max(line_map.keys(), default=0)
    new_lines = [line_map.get(i + 1, '') for i in range(max_line)]

    if new_lines:
        new_lines[0] = new_lines[0].lstrip()

    return '\n'.join(new_lines)

# test = """ codle= "코들"
# print(codle)
# """
# test_cleaned = strip_prompts(test)

"""
- WA: Wrong Answer
- TLE: Time Limit Exceeded
- AC: Accepted
- OLE: Output Limit Exceeded
- RE: Runtime Error
- IE: Internal Error

WA, TLE, AC, OLE 인 경우는 문법 오류가 없다.
"""
def preprocess_ipython(main_df:pd.DataFrame, code_df:pd.DataFrame, all=False):
    if all:
        code_df['cleaned_code'] = code_df['code'].apply(strip_prompts)
    else:
        if "error_detail" in main_df.columns:
            part_ids = main_df[main_df.error_detail == ''].code_id
        elif "verdict" in main_df.columns:
            part_ids = main_df[main_df.verdict.isin(["WA", "TLE", "AC", "OLE"])].code_id
        success_interaction = code_df.code_id.isin(part_ids)
        code_df['cleaned_code'] = code_df['code'].copy()
        code_df[success_interaction]['cleaned_code'] = code_df[success_interaction]['code'].apply(strip_prompts)

    has_prompt = code_df['cleaned_code'] != code_df['code']
    has_prompt_df = code_df[has_prompt]

    # cmp_dir = os.path.join("has_prompt_codes", f"{data_ver.name}_{dataset_type.name}")
    # cmp_orginal_dir = os.path.join(cmp_dir, "orginal")
    # cmp_cleaned_dir = os.path.join(cmp_dir, "cleaned")
    # if os.path.isdir(cmp_orginal_dir):
    #     shutil.rmtree(cmp_orginal_dir)
    # os.makedirs(cmp_orginal_dir)
    # if os.path.isdir(cmp_cleaned_dir):
    #     shutil.rmtree(cmp_cleaned_dir)
    # os.makedirs(cmp_cleaned_dir)

    # for idx, row in has_prompt_df.iterrows():
    #     with open(os.path.join(cmp_orginal_dir, f"{row.student_id}_{row.problem_id}_{idx}.txt"), "w") as f:
    #         f.write(row.code)
    #     with open(os.path.join(cmp_cleaned_dir, f"{row.student_id}_{row.problem_id}_{idx}.txt"), "w") as f:
    #         f.write(row.cleaned_code)
    # print(f"preprocess_ipython.end iPyton prompt: {len(has_prompt_df)}")
    return code_df, len(has_prompt_df)

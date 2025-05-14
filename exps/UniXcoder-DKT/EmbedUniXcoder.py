import torch
import re
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tree_sitter import Language, Parser

device = "cuda"

# Load tokenizer & model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
model = RobertaModel.from_pretrained("microsoft/unixcoder-base").to(device)
model.eval()

def remove_comments(code: str) -> str:
    code = re.sub(r'#.*', '', code)  # inline comments
    code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)  # triple double quotes
    code = re.sub(r"\'\'\'[\s\S]*?\'\'\'", '', code)  # triple single quotes
    return code.strip()

def get_tokens(code:str):
    return tokenizer.tokenize(code)

def tokenize(code, max_length=1024):
    """ 
    Convert string to token ids 
            
    Parameters:

    * `inputs`- list of input strings.
    * `max_length`- The maximum total source sequence length after tokenization.
    """
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)
    org_token_len = len(code_tokens)

    code_tokens = code_tokens[:max_length-4]
    source_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]

    tokens_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    padding_length = max_length - len(tokens_ids)
    tokens_ids += [tokenizer.pad_token_id]*padding_length

    return tokens_ids, org_token_len

def get_code_embedding(code:str) -> torch.Tensor:
    code=remove_comments(code)

    tokens_ids, org_token_len = tokenize(code)
    cur_token_len = len(tokens_ids)

    tokens_ids = torch.tensor([tokens_ids]).to(device)
    with torch.no_grad():
        outputs = model(tokens_ids,attention_mask=tokens_ids.ne(tokenizer.pad_token_id))
    outputs = outputs[0]
    outputs = (outputs*tokens_ids.ne(tokenizer.pad_token_id)[:,:,None]).sum(1)/tokens_ids.ne(tokenizer.pad_token_id).sum(-1)[:,None]
    embedding = torch.nn.functional.normalize(outputs, p=2, dim=1).cpu().detach().numpy()
    embedding = embedding[0]
    return embedding, org_token_len, cur_token_len


# def chunk_code_tokens(code, max_length=512, stride=384):
#     token_ids = tokenizer(
#         code,
#         return_tensors="pt",
#         add_special_tokens=False
#     )["input_ids"][0].to(device)

#     chunks = []
#     seq_len = token_ids.size(0)
#     step = max_length - stride
    
#     for start in range(0, seq_len, step):
#         end = min(start + max_length - 2, seq_len)  # CLS, SEP 토큰 자리 확보
#         chunk_ids = token_ids[start:end]
        
#         # [CLS] + chunk + [SEP]
#         chunk = torch.cat([
#             torch.tensor([tokenizer.cls_token_id], device=device),
#             chunk_ids,
#             torch.tensor([tokenizer.sep_token_id], device=device)
#         ])
#         # 패딩
#         if chunk.size(0) < max_length:
#             pad_len = max_length - chunk.size(0)
#             chunk = torch.cat([
#                 chunk,
#                 torch.full((pad_len,), tokenizer.pad_token_id, device=device)
#             ])
        
#         chunks.append(chunk)

#         if end == seq_len:
#             break

#     return chunks

# def get_code_embedding(code:str) -> torch.Tensor:
#     code=remove_comments(code)

#     code_chunks = chunk_code_tokens(code)
#     tokenize()
#     if len(code_chunks)>0:
#         input_ids = torch.stack(code_chunks, dim=0)           # (num_chunks, max_length)
#         attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
#         # 각 청크의 [CLS] 벡터 추출 후 평균
#         cls_embs = outputs.last_hidden_state[:, 0, :]    # (num_chunks, hidden_dim)
#         avg_emb = cls_embs.mean(dim=0)    # (1, hidden_dim)
#         return avg_emb.cpu().detach().numpy(), len(code_chunks)
#     else:
#         return np.zeros(model.config.hidden_size), 0


def get_emb_size():
    return model.config.hidden_size

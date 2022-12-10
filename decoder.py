import torch
from utilities import TextProcess

textprocess = TextProcess()

def GreedyDecode(output, blank_label=27, dim=2, collapse_repeated=True):
    arg_maxes = torch.argmax(output,dim=dim).squeeze(dim-1)
    decode = []
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and i != 0 and index == arg_maxes[i-1]:
                continue
            decode.append(index.item())
    return textprocess.int_to_text_seq(decode)

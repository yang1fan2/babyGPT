from datasets import load_dataset
import tiktoken
import numpy as np
import os
current_folder = os.path.dirname(os.path.abspath(__file__))
enc = tiktoken.get_encoding("gpt2")

def text2id(example):
    ids = enc.encode(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

for file_type in ['train', 'test', 'validation']:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=file_type)
    dataset_with_ids = dataset.map(text2id, num_proc=4, remove_columns=["text"])

    # write to binary file
    num_shards = 1024
    # save to local folder

    fp = np.memmap(os.path.join(current_folder, f"{file_type}.bn"), dtype='uint16', mode='w+', shape=(np.sum(dataset_with_ids['len']),))
    current_index = 0
    for i in range(num_shards):
        batch = dataset_with_ids.shard(num_shards=num_shards, index=i)
        ids = np.concatenate(batch["ids"])
        batch_len = np.shape(ids)[0]
        fp[current_index:current_index + batch_len] = ids
        current_index += batch_len
    fp.flush()


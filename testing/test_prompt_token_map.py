import json
import os

from toolkit.prompt_token_map import tokenize_prompt, merge_subwords_to_words, export_prompt_mapping_json


class FakeTokenizer:
    """A tiny tokenizer mock for unit tests.

    Behavior:
    - tokenizer(prompt, return_offsets_mapping=True) returns dict with input_ids and offset_mapping
    - convert_ids_to_tokens maps ids to predictable tokens
    - encode returns a list of ids
    """

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        words = text.split(" ")
        input_ids = []
        offsets = []
        pos = 0
        for w in words:
            if "suf:" in w:
                root, suf = w.split("suf:")
                input_ids.extend([len(input_ids) + 1, len(input_ids) + 2])
                offsets.append((pos, pos + len(root)))
                offsets.append((pos + len(root), pos + len(root) + len(suf)))
                pos += len(w) + 1
            else:
                input_ids.append(len(input_ids) + 1)
                offsets.append((pos, pos + len(w)))
                pos += len(w) + 1
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, ids):
        toks = []
        for i, _ in enumerate(ids):
            toks.append(f"t{i}")
        return toks

    def encode(self, text, add_special_tokens=False):
        return self(text, return_offsets_mapping=False)["input_ids"]


def test_tokenize_prompt_with_offsets():
    tok = FakeTokenizer()
    out = tokenize_prompt("hello world", tok)
    assert isinstance(out["tokens"], list)
    assert isinstance(out["ids"], list)
    assert out["offsets"] is not None
    assert len(out["tokens"]) == len(out["ids"]) == len(out["offsets"]) == 2


def test_merge_subwords_uses_offsets():
    tok = FakeTokenizer()
    # Use a prompt where the second word is split into root+subword
    enc = tok("foo suf:bar")
    # build token strings for the two tokens produced
    tokens = ["foo", "root", "##bar"]
    merged = merge_subwords_to_words(tokens, enc["offset_mapping"]) 
    # Expect two word entries
    assert len(merged) == 2
    # Second entry should group token indices [1,2]
    assert merged[1]["token_indices"] == [1, 2]


def test_merge_subwords_heuristic_no_offsets():
    tokens = ["▁hello", "##world", "plain"]
    merged = merge_subwords_to_words(tokens, offsets=None)
    # Expect two entries: one grouping the first two tokens, and one for 'plain'
    assert len(merged) == 2
    assert merged[0]["token_indices"] == [0, 1]
    assert merged[1]["token_indices"] == [2]


def test_export_prompt_mapping_json_writes_file(tmp_path):
    tok = FakeTokenizer()
    prompt = "a b c"
    out_file = tmp_path / "mapping.json"
    res = export_prompt_mapping_json(prompt, tok, safetensor_paths=[], out_path=str(out_file))
    # verify returned structure and written file
    assert os.path.exists(str(out_file))
    with open(str(out_file), "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "prompt" in loaded
    assert loaded["prompt"] == prompt
    assert "words" in loaded
    assert isinstance(loaded["words"], list)

from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("prompt_files")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def hps_v2_all():
    return from_file("hps_v2_all.txt")

def simple_animals():
    return from_file("simple_animals.txt")

def test_human_1():
    return from_file("test_human_1.txt")

def test_human_2():
    return from_file("test_human_2.txt")

def humans_20():
    return from_file("humans_20.txt")

def humans_100():
    return from_file("humans_100.txt")

def humans_500():
    return from_file("humans_500/humans_500.txt")

def humans_3000():
    return from_file("humans_3000/humans_3000.txt")

def humans_100_2():
    return from_file("humans_100/humans_100.txt")

def words_250():
    return from_file("words_250/words_250.txt")

def animals_train():
    return from_file("animals_train.txt")

def animals_eval():
    return from_file("animals_eval.txt")

def words_100():
    return from_file("words_100/words_100.txt")

def elon_train():
    return from_file("elon_train.txt")

def elon_test():
    return from_file("elon_test.txt")

def eval_simple_animals():
    return from_file("eval_simple_animals.txt")

def eval_hps_v2_all():
    return from_file("hps_v2_all_eval.txt")

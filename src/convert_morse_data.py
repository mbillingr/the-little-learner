
# convert the data file (https://github.com/themetaschemer/malt/blob/main/examples/morse/data/morse-data.rkt)

import re

def load():
    with open("../../Downloads/morse-data.rkt") as f:
        return f.read()

def tokenize(src):
    return Peekable(
        filter(
            lambda t:t, 
            re.split("([()])|\s|#.*?\n", src)))

class Peekable:
    def __init__(self, it):
        self.it = it
        self.upcoming = None
    
    def peek(self):
        if self.upcoming is None:
            self.upcoming = next(self.it)
        return self.upcoming

    def __next__(self):
        if self.upcoming is not None:
            item = self.upcoming
            self.upcoming = None
            return item
        return next(self.it)

    def __iter__(self): return self
        

def parse(tokens):
    while True:
        try:
            expr = parse_expr(tokens)
        except StopIteration:
            return
        yield expr

def parse_expr(tokens):
    t = next(tokens)
    match t:
        case "(": return parse_list(tokens)
        case _ :
            return t

def parse_list(tokens):
    the_list = []
    while True:
        if tokens.peek() == ")":
            next(tokens)
            return the_list
        the_list.append(parse_expr(tokens))

def convert(ctx):
    def convert(exp):
        match exp:
            case ["require" | "provide", *_]:
                raise Ignored()
            case ["define", name, ["list", *items]]:
                return define_dataset(name, items)
            case ["define", *_]:
                raise Ignored()
            case [first, *_]: raise NotImplementedError(f"[{first}, ...]")
            case _: raise NotImplementedError(exp)
    return convert

def define_dataset(name, items):
    xs, ys = [], []
    for item in items:
        match item:
            case ["cons", ["tensor", *x], ["tensor", *y]]:
                xs.append(list(map(float, x)))
                ys.append(list(map(float, y)))
            case _: raise NotImplementedError(item)
    return (
        f"{kebab_to_snake(name)}_xs = {xs}\n"
        f"{kebab_to_snake(name)}_ys = {ys}\n")

def kebab_to_snake(name):
    return name.replace("-", "_")

def map_ignored(f, it):
    def wrapped(*args):
        try:
            return f(*args)
        except Ignored as e:
            return e

    return filter(
        lambda x: not isinstance(x, Ignored), 
        map(wrapped, it))


class Ignored(Exception): pass

with open("src/MorseDataset.jl", "w") as f:
    f.write("module MorseDataset\n")
    f.write("export morse_train_xs, morse_train_ys, morse_test_xs, morse_test_ys, morse_validate_xs, morse_validate_ys\n")
    f.write("using ..Tensors\n")
    f.write("\n")
    for stmt in map_ignored(convert({}), parse(tokenize(load()))):
        f.write(stmt)
    f.write("\n")
    f.write("morse_train_xs = tensor(morse_ds_xs)\n")
    f.write("morse_train_ys = tensor(morse_ds_ys)\n")
    f.write("morse_test_xs = tensor(morse_test_ds_xs)\n")
    f.write("morse_test_ys = tensor(morse_test_ds_ys)\n")
    f.write("morse_validate_xs = tensor(morse_validate_ds_xs)\n")
    f.write("morse_validate_ys = tensor(morse_validate_ds_ys)\n")
    f.write("end\n")

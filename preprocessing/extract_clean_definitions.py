import re
import argparse

import pandas as pd


def convert_header_to_definition(header):
    print(header)
    match re.findall(r"<b>[\"\s]*(.*?)[\"\s]*</?b>", header):
        case [x]:
            return x
        case [x, y]:
            return x + y
        case _:
            return None

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input")
    arg_parser.add_argument("--output")
    args = arg_parser.parse_args()

    raw_contracts = pd.read_csv(args.input, index_col = ["Title"])
    raw_contracts["definition"] = raw_contracts.header.apply(convert_header_to_definition)

    raw_contracts.to_csv(args.output)

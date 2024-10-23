#!/usr/bin/env python
import glob
import os
import sys

HEADER = '''/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

'''


def add_header(file):
    with open(file) as f:
        contents = f.readlines()

    # do nothing if there is already header
    if contents and contents[0].startswith('/*-'):
        return

    with open(file, 'w') as out:
        out.write(HEADER)
        out.writelines(contents)


if __name__ == '__main__':
    dir = sys.argv[1]
    if not os.path.isdir(dir):
        raise ValueError(f'{dir} is not a directory')

    for file in glob.glob(dir + '/*.ts'):
        add_header(file)

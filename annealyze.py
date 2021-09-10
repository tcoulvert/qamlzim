#!/usr/bin/env python
import subprocess

def git_hash():
    hash = subprocess.check_output('git rev-parse HEAD', shell=True).decode('utf-8')
    return hash.strip()

print(git_hash())

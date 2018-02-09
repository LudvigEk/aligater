#!/usr/bin/env bash
jupyter nbconvert $1 --to python --template=strip_markdown.tpl

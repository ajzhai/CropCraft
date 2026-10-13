#!/usr/bin/env python3
"""Wrapper to modify ggssvt.config.MODEL before running ggssvt.campaign.

Usage: python tools/run_campaign_wrapper.py [args for ggssvt.campaign]
Set environment variable GGSSVT_QUERY_CHUNK to an integer to override MODEL.query_chunk.
"""
import os
import sys
import dataclasses

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Read desired chunk from env
chunk_env = os.environ.get("GGSSVT_QUERY_CHUNK")
if chunk_env:
    try:
        chunk = int(chunk_env)
    except Exception:
        print(f"Invalid GGSSVT_QUERY_CHUNK='{chunk_env}'; must be int", file=sys.stderr)
        chunk = None
else:
    chunk = None

if chunk is not None:
    # Import config and replace MODEL before ggssvt.campaign imports it
    import ggssvt.config as _config

    try:
        _config.MODEL = dataclasses.replace(_config.MODEL, query_chunk=chunk)
        print(f"Set ggssvt.config.MODEL.query_chunk = {chunk}")
    except Exception as e:
        print(f"Failed to set MODEL.query_chunk: {e}", file=sys.stderr)

# Forward args to ggssvt.campaign
# The target module expects to be run as `python -m ggssvt.campaign ...`
# We'll set sys.argv accordingly and run the module.
import runpy

# Build argv for the target module
sys.argv = ["ggssvt.campaign"] + sys.argv[1:]

# Run the module as __main__
runpy.run_module("ggssvt.campaign", run_name="__main__")

#!/usr/bin/env python3
"""Run any of the four official benchmark entry points from the pinned snapshot."""
import argparse
import importlib
import sys
from core import UPSTREAM, encode_qwen25, whole_qwen25_frame_tokens, verify_upstream


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--benchmark', choices=['ovobench','streamingbench'], required=True)
    p.add_argument('--mode', choices=['reactive','proactive'], required=True)
    p.add_argument('--model', choices=['qwen3','qwen25'], default='qwen3')
    a, rest = p.parse_known_args()
    verify_upstream()
    import transformers
    if transformers.__version__!='4.57.1': raise RuntimeError('Requires transformers==4.57.1')
    sys.path.insert(0,str(UPSTREAM))
    module=importlib.import_module(f'eval_{a.benchmark}_{a.mode}')
    if a.model=='qwen25':
        module.Qwen3VLForConditionalGeneration=transformers.Qwen2_5_VLForConditionalGeneration
        module.encode_images=encode_qwen25
        module.whole_current_frame_tokens=whole_qwen25_frame_tokens
    sys.argv=[str(UPSTREAM/f'eval_{a.benchmark}_{a.mode}.py'),*rest]
    module.main()


if __name__=='__main__': main()

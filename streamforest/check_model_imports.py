"""Check the official model import chain without loading weights or requiring GPUs."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
import sys


def import_with_original_error(module_name, catch_file):
    """Expose exceptions swallowed by upstream's model registration __init__.py.

    Trace only that file for the duration of this import. Do not edit official
    sources, replace modules or suppress missing dependencies.
    """
    caught = []
    catch_files = {str(Path(catch_file).absolute()), str(Path(catch_file).resolve())}
    previous = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code.co_filename in catch_files:
            if event == 'exception':
                caught.append(arg)
            return trace
        return None

    try:
        sys.settrace(trace)
        return importlib.import_module(module_name)
    except Exception as outer:
        if caught:
            _, original, tb = caught[0]
            raise RuntimeError(
                f'Official model registration failed: {type(original).__name__}: {original}. '
                f'The later {type(outer).__name__}: {outer} is a secondary error.'
            ) from original.with_traceback(tb)
        raise
    finally:
        sys.settrace(previous)


def load_builder(root):
    root = Path(root).expanduser().resolve()
    sys.path.insert(0, str(root))
    return import_with_original_error('llava.model.builder', root / 'llava/model/__init__.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--streamforest-root', type=Path,
                        default=Path(__file__).resolve().parent / 'StreamForest')
    args = parser.parse_args()
    print(f'Python: {sys.executable}', flush=True)
    for name in ['torch', 'torchvision', 'transformers', 'accelerate', 'numpy',
                 'flash-attn', 'deepspeed', 'timm', 'einops', 'av', 'decord']:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = 'NOT INSTALLED'
        print(f'{name}: {version}', flush=True)
    from streamforest_backend import verify_upstream
    verify_upstream(args.streamforest_root)
    module = load_builder(args.streamforest_root)
    print(f'Model imports passed: {module.__file__}', flush=True)
    print('Weights, CUDA execution and inference have NOT been tested.', flush=True)


if __name__ == '__main__':
    main()

"""Import pinned official packages by file path, independently of generic module names."""
import importlib.util
from pathlib import Path
import sys


def validate_source(root):
    root = Path(root).expanduser().resolve() / 'Flash-VStream-Qwen'
    required = ('models/__init__.py', 'models/vstream_qwen2vl_model.py',
                'models/vstream_qwen2vl_processor.py', 'models/compress_functions.py',
                'models/flash_memory_constants.py', 'qwen_vl_utils/__init__.py',
                'qwen_vl_utils/vision_process.py')
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError('Official source is incomplete or --upstream-root points to the wrong directory. '
                                'Restore the pinned upstream snapshot (not the checkpoint directory). Missing:\n'
                                + '\n'.join(missing))
    return root


def load_package(name, folder):
    source = (Path(folder) / '__init__.py').resolve()
    if name in sys.modules:
        existing = sys.modules[name]
        if getattr(existing, '__file__', None) == str(source):
            return existing
        raise RuntimeError(f'Official package alias {name} already refers to a different source')
    spec = importlib.util.spec_from_file_location(name, source,
                                                 submodule_search_locations=[str(source.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + '.'):
                del sys.modules[key]
        raise
    return module


def load_official(root):
    source = validate_source(root)
    models = load_package('_flash_vstream_official_models', source / 'models')
    vision = load_package('_flash_vstream_official_vision', source / 'qwen_vl_utils')
    return models, vision

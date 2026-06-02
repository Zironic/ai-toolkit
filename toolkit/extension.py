import os
import importlib
import pkgutil
from typing import List

from toolkit.paths import TOOLKIT_ROOT


class Extension(object):
    """Base class for extensions.

    Extensions are registered with the ExtensionManager, which is
    responsible for calling the extension's load() and unload()
    methods at the appropriate times.

    """

    name: str = None
    uid: str = None

    @classmethod
    def get_process(cls):
        # extend in subclass
        pass


def get_all_extensions_process_dict(needed_types: set = None) -> dict:
    """Build a uid -> process-class dict, importing only what is needed.

    Parameters
    ----------
    needed_types:
        Set of process-type strings required by the current job.  When
        provided the loader stops as soon as every type has been found,
        skipping unrelated (and potentially heavy) extension modules
        entirely.  Pass ``None`` to load everything (legacy behaviour).
    """
    extension_folders = ['extensions', 'extensions_built_in']
    process_dict = {}
    remaining = set(needed_types) if needed_types is not None else None

    for sub_dir in extension_folders:
        if remaining is not None and not remaining:
            break  # all needed types already found
        extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
        for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
            if remaining is not None and not remaining:
                break  # all needed types already found
            try:
                module = importlib.import_module(f"{sub_dir}.{name}")
                extensions = getattr(module, "AI_TOOLKIT_EXTENSIONS", None)
                if isinstance(extensions, list):
                    for ext in extensions:
                        if remaining is None or ext.uid in remaining:
                            process_dict[ext.uid] = ext.get_process()
                            if remaining is not None:
                                remaining.discard(ext.uid)
            except ImportError as e:
                print(f"Warning: skipped extension '{name}' (missing dependency: {e})")

    return process_dict

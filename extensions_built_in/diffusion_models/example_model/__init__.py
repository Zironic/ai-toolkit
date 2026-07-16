# This is a documentation-only TEMPLATE model. Start with README.md in this
# folder for the full guide to adding a new model architecture to ai-toolkit.
#
# It is intentionally NOT registered: the parent package
# (extensions_built_in/diffusion_models/__init__.py) does not import it, so it
# never shows up as a trainable arch. To register a real model, import its
# class there and append it to the AI_TOOLKIT_MODELS list. (Models can also
# live in their own folder under extensions/, which defines its own
# module-level AI_TOOLKIT_MODELS list -- the scan in
# toolkit/util/get_model.py:get_all_models() treats extensions/ and
# extensions_built_in/ identically.)
from .example_model import ExampleModel

__all__ = ["ExampleModel"]

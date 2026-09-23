"""Strict YAML loading for the coefficient registry.

The registry's whole point is that a number states where it came from and where it
holds; a silent parse would undo that. So loading is strict in three ways the stock
``yaml.safe_load`` is not:

  * duplicate mapping keys are an error, not last-wins — a second ``value:`` in an
    entry must never silently shadow the first;
  * a complex (unhashable) mapping key — a list or map used as a key — is a named
    error rather than a raw ``TypeError`` escaping to CI;
  * the loader is only ever asked for ``dict``/``list``/scalar data (SafeLoader),
    never arbitrary Python objects.

Everything else (unknown keys, wrong enums) is caught by the schema layer, not here.
"""

from __future__ import annotations

import yaml


class DuplicateKeyError(ValueError):
    """Raised for a repeated mapping key, or an unsupported complex (unhashable) key."""


class _StrictLoader(yaml.SafeLoader):
    """A SafeLoader that refuses duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        # A complex YAML key (a list or map used as a key) is unhashable and has no place
        # in this schema. Reject it as a named error rather than let `key in mapping`
        # crash with a raw TypeError that would escape to CI.
        try:
            duplicate = key in mapping
        except TypeError:
            mark = key_node.start_mark
            raise DuplicateKeyError(
                f"unsupported complex mapping key {key!r} at line {mark.line + 1}, "
                f"column {mark.column + 1} (keys must be scalars)"
            )
        if duplicate:
            mark = key_node.start_mark
            raise DuplicateKeyError(
                f"duplicate key {key!r} at line {mark.line + 1}, column {mark.column + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_strict(text: str):
    """Parse a single YAML document with duplicate-key detection.

    Returns the parsed value (``None`` for an empty document). Raises
    ``DuplicateKeyError`` on a repeated mapping key or an unsupported complex
    (unhashable) key, and ``yaml.YAMLError`` on any other malformed input.
    """
    return yaml.load(text, Loader=_StrictLoader)

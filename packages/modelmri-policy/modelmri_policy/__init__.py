"""Hold a robot policy in its own process, so ModelMRI can ask what it would do.

Separate from `modelmri` because lerobot pins torch and numpy hard enough that
sharing an environment breaks both. See `contract.py` for the wire format and
why its version is declared on each side independently.
"""

from .contract import CONTRACT, ContractError, check

__all__ = ["CONTRACT", "ContractError", "check"]
__version__ = "0.1.0"

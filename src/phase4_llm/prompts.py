"""Forward to the single phase4_llm implementation."""
import sys
from phase4_llm import prompts as _impl
sys.modules[__name__] = _impl

"""Forward to the single phase4_llm implementation."""
import sys
from phase4_llm import comparison_schema as _impl
sys.modules[__name__] = _impl

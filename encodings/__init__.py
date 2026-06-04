from .registry import register_encoding, get_encoding_cls, list_encodings
from . import havlicek  
from . import angle     
from . import zzprod    
from . import amplitude 
from . import basis     
from . import denseangle 

# Phase-2: additional encodings from the thesis "Encoding Literature Tree"
from . import histogram
from . import sparse_amplitude
from . import reuploading
from . import hamiltonian
from . import trainable_kernel
from . import integer
from . import onehot

try:
    from . import hardware_aware
except ModuleNotFoundError as exc:
    if "hardware_aware" not in str(exc):
        raise

try:
    from . import ha_sage
except ModuleNotFoundError as exc:
    if "ha_sage" not in str(exc):
        raise

try:
    from . import ha_sage_cmtsd
except ModuleNotFoundError as exc:
    if "ha_sage_cmtsd" not in str(exc):
        raise

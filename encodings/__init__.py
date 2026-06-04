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
from . import hardware_aware
from . import ha_sage
from . import ha_sage_cmtsd

from .bert_cls.bert_cls import BERT_CLS
from .scibert_cls.scibert_cls import SciBERT_CLS
from .scibert_concat.scibert_concat import SciBERT_Concat
from .scibert_max.scibert_max import SciBERT_Max
from .scibert_gate.scibert_gate import SciBERT_Gate
from .bert_gcn.bert_gcn import BERT_GCN
from .bert_mhan.bert_mhan import BERT_MHAN
from .transformer.transformer import Transformer
from .textgcn.textgcn import TextGCN
from .igcn_bert.igcn_bert import IGCNBERT

try:
    from .dgc_bert.dgcbert import DGCBERT
    _dgc_available = True
except ImportError:
    DGCBERT = None
    _dgc_available = False

__all__ = ["BERT_CLS", "SciBERT_CLS", "SciBERT_Concat", "SciBERT_Max", "SciBERT_Gate", "BERT_GCN", "BERT_MHAN", "Transformer", "TextGCN", "IGCNBERT", "DGCBERT"]

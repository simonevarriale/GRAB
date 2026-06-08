from src.dataset.hctqa import HCTQADataset
from src.dataset.wtq import WTQDatasetOrig
from src.dataset.wikisql import WikiSQLDataset
from src.dataset.structProbe import StructProbeDataset
from src.dataset.hitab import HiTabDataset
from src.dataset.tabfact import TabFactDataset
from src.dataset.tabmwp import TabMWPDataset
from src.dataset.multihiertt import MultiHierTTDataset
from src.dataset.scitat import SciTabDataset
from src.dataset.mmqa import MMQADataset
from src.dataset.tqa_bench import TQABenchDataset
from src.dataset.atis import ATISDataset
from src.dataset.geoquery import GeoQueryDataset
from src.dataset.spider_text2sql import SpiderDataset
from src.dataset.spider_qa import SpiderQADataset

load_dataset = {
    'wtq':         WTQDatasetOrig,
    'wikisql':     WikiSQLDataset,
    'structProbe': StructProbeDataset,
    'hitab':       HiTabDataset,
    'hctqa':       HCTQADataset,
    'tabfact':     TabFactDataset,
    'tabmwp':      TabMWPDataset,
    'multihiertt': MultiHierTTDataset,
    'scitat':      SciTabDataset,
    'mmqa':        MMQADataset,
    'tqa_bench':   TQABenchDataset,
    'atis':        ATISDataset,
    'geoquery':    GeoQueryDataset,
    'spider_text2sql': SpiderDataset,
    'spider_qa':   SpiderQADataset,
}

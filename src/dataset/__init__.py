from src.dataset.hctqa import HCTQADataset
from src.dataset.hctqa_stress_test import HCTQAStressTestDataset
from src.dataset.wtq import WTQDatasetOrig
from src.dataset.wikisql import WikiSQLDataset
from src.dataset.structProbe import StructProbeDataset
from src.dataset.hitab import HiTabDataset
from src.dataset.tabfact import TabFactDataset
from src.dataset.tabmwp import TabMWPDataset
from src.dataset.multihiertt import MultiHierTTDataset
from src.dataset.scitat import SciTabDataset
from src.dataset.mmqa import MMQADataset
from src.dataset.tqa_bench import (TQABenchDataset, TQABench16kDataset,
    TQABench32kDataset, TQABench64kDataset, TQABench128kDataset)
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
    'hctqa_stress_test': HCTQAStressTestDataset,
    'tabfact':     TabFactDataset,
    'tabmwp':      TabMWPDataset,
    'multihiertt': MultiHierTTDataset,
    'scitat':      SciTabDataset,
    'mmqa':        MMQADataset,
    'tqa_bench':   TQABenchDataset,
    'tqa_bench_16k': TQABench16kDataset,
    'tqa_bench_32k': TQABench32kDataset,
    'tqa_bench_64k': TQABench64kDataset,
    'tqa_bench_128k': TQABench128kDataset,
    'atis':        ATISDataset,
    'geoquery':    GeoQueryDataset,
    'spider_text2sql': SpiderDataset,
    'spider_qa':   SpiderQADataset,
}

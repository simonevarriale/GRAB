import random

from src.dataset.sql_result_dataset import SQLResultDataset

_SPIDER_SEED = 42
_SPIDER_VAL_FRAC = 0.1


class SpiderQADataset(SQLResultDataset):
    """Spider text-to-SQL dataset (scratch format) recast as result-table generation.

    Uses the same data format as ATISDataset / GeoQueryDataset (answer field
    is a JSON result table; target is the linearized result table) and evaluates
    with table EM metrics rather than denotation accuracy.

    Split strategy (Spider has no public test set):
      - train / validation : deterministic 90/10 carve of the HF train split
                             (seed=42, consistent with other re-split datasets)
      - test               : the official HF validation (dev) split, never seen
                             during training or model selection
    """

    def __init__(self, type, **kwargs):
        if type == 'test':
            # Official HF dev set used purely for held-out evaluation
            super().__init__('validation', dataset_name='spider', **kwargs)
            self._id_map = list(range(len(self.data)))
            self.precomputed_split = 'validation'
            print('Spider: test split = HF validation (held-out, not used for model selection)')
        elif type in ('train', 'validation'):
            # Load full HF train then carve
            super().__init__('train', dataset_name='spider', **kwargs)
            n = len(self.data)
            rng = random.Random(_SPIDER_SEED)
            idx = list(range(n))
            rng.shuffle(idx)
            n_val = max(1, int(n * _SPIDER_VAL_FRAC))
            subset = sorted(idx[:n_val] if type == 'validation' else idx[n_val:])
            # _id_map[new_pos] = original HF train index — keeps .pt filenames stable
            self._id_map = subset
            self.data = self.data.select(subset)
            # Both train and validation are carved from HF train → .pt files live in train/
            self.precomputed_split = 'train'
            print(f'Spider re-split ({type}): {len(self.data)} samples '
                  f'(90/10 carve of HF train, seed={_SPIDER_SEED})')
        else:
            super().__init__(type, dataset_name='spider', **kwargs)
            self._id_map = list(range(len(self.data)))
            self.precomputed_split = type
        self.type = type

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # Use the original HF train index as id so PrecomputedGraphDataset
        # resolves the correct {id}.pt file regardless of how the split is carved.
        sample['id'] = self._id_map[index]
        return sample

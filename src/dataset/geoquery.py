from src.dataset.sql_result_dataset import SQLResultDataset


class GeoQueryDataset(SQLResultDataset):
    """GeoQuery text-to-SQL dataset recast as result-table generation."""

    def __init__(self, type, **kwargs):
        super().__init__(type, dataset_name='geoquery', **kwargs)

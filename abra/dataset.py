#!/usr/bin/python
# -*- coding: utf-8 -*-
from abra.config import search_config
from pandas import DataFrame


class DatasetException(Exception):
    pass


class Dataset(object):
    """
    Interface between raw data, global configuration, and experiment
    """
    def __init__(self, df, treatment=None, measures=None, attributes=None, meta=None):
        treatment = treatment if treatment else search_config(df, "experiment", "treatment")[0]   # always first
        if treatment is None:
            raise DatasetException(f'treatment column {treatment} not in dataframe')
        else:
            self.treatment = treatment

        if isinstance(measures, str):
            measures = [measures]

        if isinstance(meta, str):
            meta = [meta]

        self.meta = meta if meta else []
        self.measures = measures if measures \
            else search_config(df, "experiment", "measures")
        self.attributes = attributes if attributes \
            else search_config(df, "experiment", "attributes")

        all_columns = [self.treatment] + self.measures + self.attributes + self.meta
        self.data = df[all_columns]
        self.columns = set(all_columns)

    def __repr__(self):
        return f"Dataset(measures={self.measures}, attributes={self.attributes})"

    @property
    def cohorts(self):
        """
        Return a list of cohorts defined by the experiment treatment.
        """
        if not hasattr(self, '_cohorts'):
            self._cohorts = sorted(self.data[self.treatment].unique().tolist())
        return self._cohorts

    def segments(self, attribute):
        """
        Return a list of tuples containing (treatment, segment) pairs, for a
        given segmentation attribute.
        """
        return self.data.groupby([self.treatment, attribute]).sum().index.tolist()

    @property
    def cohort_measures(self):
        """
        Return metric samples for each cohort.
        """
        measures = {}
        for cohort in self.cohorts:
            measures[cohort] = {}
            for metric in self.measures:
                mask = self.data[self.treatment] == cohort
                measures[cohort][metric] = self.data[mask][metric].values

        return DataFrame(measures).T

    def segment_samples(self, attribute):
        """
        Return samples from each segment (treatment-attribute pair).
        """
        measures = {}
        for segment in self.segments(attribute):
            measures[segment] = {}
            for metric in self.measures:
                mask = (self.data[self.treatment] == segment[0]) \
                    & (self.data[attribute] == segment[1])
                measures[segment][metric] = self.data[mask][metric].values
        return DataFrame(measures).T

import pytest
from abra.dataset import Dataset, DataFrame


def test_default_init(proportions_data_small):
    dataset = Dataset(proportions_data_small)
    # default configuration template
    assert dataset.treatment == 'treatment'
    assert 'metric' in dataset.measures
    assert dataset.__repr__() == "Dataset(measures=['metric'], attributes=['attr_0', 'attr_1'])"


def test_properties(proportions_data_small):
    dataset = Dataset(proportions_data_small)

    assert 'A' in dataset.cohorts
    assert 'B' in dataset.cohorts
    cohort_measures = dataset.cohort_measures
    assert isinstance(cohort_measures, DataFrame)
    assert cohort_measures.shape[0] == len(dataset.cohorts)


def test_segments(proportions_data_small):
    dataset = Dataset(
        df=proportions_data_small,
        attributes=['attr_0', 'attr_1']
    )

    segments = dataset.segments('attr_0')
    assert isinstance(segments, list)
    assert ('A', 'A0a') in segments
    assert ('F', 'A0b') in segments

    segment_samples = dataset.segment_samples('attr_0')
    assert isinstance(segment_samples, DataFrame)
    assert ('A', 'A0a') in segment_samples.index
    assert ('F', 'A0b') in segment_samples.index

    # Replace column names with spaced column names, ensure that
    # segmentation based off pandas.query still works as exptected
    proportions_data_small.columns = [c.replace("_", " ")
        for c in proportions_data_small.columns]
    dataset = Dataset(
        df=proportions_data_small,
        attributes=['attr 0', 'attr 1']
    )

    segments = dataset.segments('attr 0')
    assert isinstance(segments, list)
    assert ('A', 'A0a') in segments
    assert ('F', 'A0b') in segments

    segment_samples = dataset.segment_samples('attr 0')
    assert isinstance(segment_samples, DataFrame)
    assert ('A', 'A0a') in segment_samples.index
    assert ('F', 'A0b') in segment_samples.index

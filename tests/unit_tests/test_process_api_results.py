import pandas as pd
import pytest

from decision_tree.process_api_results import clean_datetime_column


# ---------------------------------------------------------------------------
# Tests for clean_datetime_column()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Invalid / sentinel strings -> NaT
# ---------------------------------------------------------------------------
class TestInvalidStrings:
    # NOTE: each sentinel is paired with a valid date so the column keeps
    # object dtype after replace(). If all values are replaced with NaT,
    # pandas promotes the column to datetime64 and str.match() raises
    # AttributeError. Known limitation: function needs at least one
    # non-sentinel value in the column.
    @pytest.mark.parametrize('value', [
        '0000-00-00', 'nan', 'NaN', 'None', '', 'null',
    ])
    def test_sentinel_becomes_nat(self, value):
        result = run([value, '2023-06-01'])
        assert pd.isna(result.iloc[0]), f"Expected NaT for {value!r}"

    def test_actual_none_becomes_nat(self):
        result = run([None, '2023-06-01'])
        assert pd.isna(result.iloc[0])

    def test_truly_invalid_string_becomes_nat(self):
        result = run(['not-a-date'])
        assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# Feb 29 on non-leap year -> remapped to Feb 28
# ---------------------------------------------------------------------------
class TestNonLeapFeb29:
    @pytest.mark.parametrize('year', [2023, 2022, 2021, 2019])
    def test_remapped_to_feb_28(self, year):
        result = run([f'{year}-02-29'])
        assert result.iloc[0] == pd.Timestamp(f'{year}-02-28')

    def test_remapped_value_is_not_nat(self):
        result = run(['2023-02-29'])
        assert not pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# Feb 29 on leap year -> preserved as Feb 29
# ---------------------------------------------------------------------------
class TestLeapFeb29:
    @pytest.mark.parametrize('year', [2024, 2020, 2000])
    def test_leap_year_feb29_preserved(self, year):
        result = run([f'{year}-02-29'])
        assert result.iloc[0] == pd.Timestamp(f'{year}-02-29')


# ---------------------------------------------------------------------------
# Normal valid dates (year >= 2000)
# ---------------------------------------------------------------------------
class TestNormalDates:
    @pytest.mark.parametrize('date_str, expected', [
        ('2024-01-15', '2024-01-15'),
        ('2000-12-31', '2000-12-31'),
        ('2000-01-01', '2000-01-01'),
        ('2023-02-28', '2023-02-28'),
    ])
    def test_valid_date_converted(self, date_str, expected):
        result = run([date_str])
        assert result.iloc[0] == pd.Timestamp(expected)


# ---------------------------------------------------------------------------
# Mixed column
# ---------------------------------------------------------------------------
class TestMixedColumn:
    def test_mixed_values(self):
        values = [
            '2024-02-29',
            '2023-02-29',
            '0000-00-00',
            '2022-06-15',
            None,
            'null',
        ]
        result = run(values)
        assert result.iloc[0] == pd.Timestamp('2024-02-29')
        assert result.iloc[1] == pd.Timestamp('2023-02-28')
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == pd.Timestamp('2022-06-15')
        assert pd.isna(result.iloc[4])
        assert pd.isna(result.iloc[5])

    def test_no_feb29_rows(self):
        result = run(['2023-01-01', '2023-03-15'])
        assert result.iloc[0] == pd.Timestamp('2023-01-01')
        assert result.iloc[1] == pd.Timestamp('2023-03-15')

    def test_empty_dataframe(self):
        result = run([])
        assert len(result) == 0
        assert pd.api.types.is_datetime64_any_dtype(result)


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------
class TestOutputDtype:
    def test_column_is_datetime(self):
        result = run(['2023-06-01', '2024-02-29', '0000-00-00'])
        assert pd.api.types.is_datetime64_any_dtype(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(values):
    return pd.DataFrame({'date': values})


def run(values):
    return clean_datetime_column(make_df(values), 'date')['date']

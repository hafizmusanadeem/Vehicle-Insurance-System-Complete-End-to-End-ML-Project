import pandas as pd
from src.components.transformers import GenderMapper, ColumnDropper

def test_gender_mapper():
    df = pd.DataFrame({"Gender": ["Male", "Female"]})
    transformer = GenderMapper()
    result = transformer.fit_transform(df)

    assert result["Gender"].tolist() == [1, 0]

def test_column_dropper():
    df = pd.DataFrame({'A': [1], 'B': [2]})
    dropper = ColumnDropper(columns=['A'])
    result = dropper.transform(df)
    
    assert 'A' not in result.columns
    assert 'B' in result.columns
    assert result.shape == (1, 1)

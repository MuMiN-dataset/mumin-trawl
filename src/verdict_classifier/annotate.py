'''Annotate a dataset for the verdict classifier'''

import pigeonXT as pixt
from dotenv import load_dotenv; load_dotenv()
import pandas as pd

from ..fetch_facts import fetch_facts
from ..utils import root_dir


def get_claim_df(include_duplicates: bool = False) -> pd.DataFrame:
    '''Get a dataframe containing all claims and verdicts.

    Returns:
        Pandas DataFrame: The dataframe containing all claims and verdicts.
    '''
    # If the fact data files do not already exist then this fetches them
    fetch_facts()

    # Set up directory with all the fact data files
    reviewer_dir = root_dir / 'data' / 'reviewers'

    # Get list of all the claim dataframes
    claim_dfs = [pd.read_csv(fname) for fname in reviewer_dir.glob('*.csv')]

    # Concatenate the claim dataframes
    claim_df = pd.concat(claim_dfs, axis=0).reset_index(drop=True)

    # Rename predicted_verdict column, if present
    if 'predicted_verdict' in claim_df.columns:
        renaming_dict = dict(predicted_verdict='old_predicted_verdict')
        claim_df = claim_df.rename(columns=renaming_dict)

    if 'predicted_verdict_confidence' in claim_df.columns:
        claim_df = (claim_df.rename(columns=renaming_dict)
                            .sort_values(by='predicted_verdict_confidence'))
        new_name = 'old_predicted_verdict_confidence'
        renaming_dict = dict(predicted_verdict_confidence=new_name)

    if not include_duplicates:

        # Remove the verdicts that have already been labelled
        labelled_path = root_dir / 'data' / 'verdict_annotations.csv'
        labelled = pd.read_csv(labelled_path).raw_verdict_en.tolist()
        claim_df = claim_df[~claim_df.raw_verdict_en.isin(labelled)]

        # Remove the test verdicts that have already been labelled
        labelled_path = root_dir / 'data' / 'verdict_annotations_test.csv'
        labelled = pd.read_csv(labelled_path).raw_verdict_en.tolist()
        claim_df = claim_df[~claim_df.raw_verdict_en.isin(labelled)]

        # Drop duplicates
        claim_df = (claim_df.drop_duplicates(subset='raw_verdict_en')
                            .reset_index(drop=True))

    else:

        # Remove the verdicts that have already been labelled
        labelled_path = root_dir / 'data' / 'verdict_annotations.csv'
        labelled = pd.read_csv(labelled_path).claim.tolist()
        claim_df = claim_df[~claim_df.claim.isin(labelled)]

        # Remove the test verdicts that have already been labelled
        labelled_path = root_dir / 'data' / 'verdict_annotations_test.csv'
        labelled = pd.read_csv(labelled_path).claim.tolist()
        claim_df = claim_df[~claim_df.claim.isin(labelled)]

        # Shuffle the dataframe
        claim_df = claim_df.sample(frac=1.0)

    return claim_df


def annotate_verdicts(claim_df: pd.DataFrame,
                      num_verdicts: int = 100) -> pd.DataFrame:
    '''Annotate plaintext verdicts into veracity classes.

    Args:
        claim_df (Pandas DataFrame):
            A dataframe containing the columns 'raw_verdict' and
            'raw_verdict_en'.
        num_verdicts (int, optional):
            How many verdicts will be annotated. Defaults to 100.

    Returns:
        Pandas DataFrame: The dataframe containing the annotated labels.
    '''
    # Set up the class labels
    options = ['factual', 'misinformation', 'other']

    if len(claim_df) > num_verdicts:
        claim_df = claim_df.iloc[:num_verdicts]

    # Convert dataframe to list of pairs of the original plaintext verdict and
    # its translation
    records = claim_df.to_records(index=False).tolist()

    def display_fn(raw_verdict_data: tuple):
        print(f'Translated verdict:\n    "{raw_verdict_data[-3]}"\n\n'
              f'Original verdict:\n    "{raw_verdict_data[-4]}"')

    return pixt.annotate(records,
                         options=options,
                         display_fn=display_fn,
                         buttons_in_a_row=5,
                         example_column='claim_data',
                         value_column='verdict')


def process_annotations(annotation_df: pd.DataFrame,
                        include_duplicates: bool = False,
                        output_fname: str = 'verdict_annotations.csv'
                        ) -> pd.DataFrame:
    '''Process the annotations and save them to disk.

    Args:
        annotation_df (pd.DataFrame):
            The annotation dataframe with columns 'claim_data', 'changed' and
            'verdict'.
        output_fname (str, optional):
            The name of the file where the annotations should be saved. The
            file will be put in the 'data' directory. If the file already
            exists then the new annotations will be appended to the existing
            annotations, with duplicates removed. Defaults to
            'verdict_annotations.csv'.
    '''
    output_path = root_dir / 'data' / output_fname

    annotation_df = (annotation_df.query('changed == True')
                                  .drop(columns='changed'))

    cols = ['claim', 'date', 'source', 'reviewer', 'language',
            'raw_verdict', 'raw_verdict_en']
    for idx, col in enumerate(cols):
        annotation_df[col] = annotation_df.claim_data.map(lambda x: x[idx])

    # Remove claim data column
    annotation_df.drop(columns='claim_data', inplace=True)

    if output_path.exists():
        old_annotations = pd.read_csv(output_path)
        annotation_df = (pd.concat((old_annotations, annotation_df), axis=0)
                           .reset_index(drop=True))

        if not include_duplicates:
            annotation_df = (annotation_df
                             .drop_duplicates(subset='raw_verdict_en')
                             .reset_index(drop=True))

    annotation_df.to_csv(output_path, index=False)

    return annotation_df

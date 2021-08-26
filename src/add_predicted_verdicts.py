'''Use the pretrained verdict classifier to add predicted verdicts to claims'''

from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from src.verdict_classifier import VerdictClassifier


def add_predicted_verdicts():
    '''Add predicted verdicts to all reviewer CSV files'''

    # Load the pretrained model, and give an error if it does not exist
    try:
        ckpt = next(Path('models').glob('*.ckpt'))
        model = VerdictClassifier.load_from_checkpoint(str(ckpt))
        model.eval()
    except StopIteration:
        raise RuntimeError('No pretrained verdict classifier in `models`'
                           'directory!')

    # Iterate over all the reviewer CSV files
    reviewer_dir = Path('data') / 'reviewers'
    desc = 'Predicting verdicts'
    for reviewer_csv in tqdm(list(reviewer_dir.iterdir()), desc=desc):

        # Load in the reviews
        reviewer_df = pd.read_csv(reviewer_csv)

        # Get the predicted verdict for every raw verdict, and store them in
        # `predicted_verdicts`
        predicted_verdicts = list()
        for _, row in reviewer_df.iterrows():
            raw_verdict = row.raw_verdict_en
            verdict = model.predict([raw_verdict]).predicted_verdict[0]
            predicted_verdicts.append(verdict)

        # Add `predicted_verdicts` to the CSV and save it
        reviewer_df['predicted_verdict'] = predicted_verdicts
        reviewer_df.to_csv(reviewer_csv, index=False)

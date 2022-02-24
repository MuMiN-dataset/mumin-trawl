'''Use the pretrained verdict classifier to add predicted verdicts to claims'''

from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline


def add_predicted_verdicts():
    '''Add predicted verdicts to all reviewer CSV files'''

    # Load the pretrained model, and give an error if it does not exist
    pipe = pipeline(task='text-classification',
                    model='saattrupdan/verdict-classifier-en')

    # Iterate over all the reviewer CSV files
    reviewer_dir = Path('data') / 'reviewers'
    desc = 'Predicting verdicts'
    for reviewer_csv in tqdm(list(reviewer_dir.iterdir()), desc=desc):

        # Load in the reviews
        reviewer_df = pd.read_csv(reviewer_csv)

        # Get the predicted verdict for every raw verdict, and store them in
        # `predicted_verdicts`
        predicted_verdicts = list()
        confidences = list()
        for _, row in reviewer_df.iterrows():
            raw_verdict = row.raw_verdict_en
            pred = pipe(raw_verdict, truncation=True)[0]
            verdict = pred['label']
            confidence = pred['score']
            predicted_verdicts.append(verdict)
            confidences.append(confidence)

        # Add `predicted_verdicts` to the CSV and save it
        reviewer_df['predicted_verdict'] = predicted_verdicts
        reviewer_df['predicted_verdict_confidence'] = confidences
        reviewer_df.to_csv(reviewer_csv, index=False)

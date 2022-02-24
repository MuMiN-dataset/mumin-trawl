'''Expand annotated verdicts by translating them into other languages'''

import pandas as pd
import yaml
from pathlib import Path
from tqdm.auto import tqdm
from src.translator import GoogleTranslator


def expand_verdicts():

    # Initialise paths
    anns_path = Path('data') / 'verdict_annotations.csv'
    new_anns_path = Path('data') / 'new_verdict_annotations.csv'

    # Load in the annotated date
    annotations = pd.read_csv(anns_path)

    # Initialise translator
    translator = GoogleTranslator()

    # Load list of languages
    language_path = Path('data') / 'bcp47.yaml'
    languages = [language
                 for language in yaml.safe_load(language_path.read_text())
                 if isinstance(language, str)]

    # Initialise data dicts
    annotations_dict = annotations.copy().to_dict('list')

    # Loop over all the rows in the annotations dataframe
    for _, row in tqdm(list(annotations.iterrows())):

        # Loop over all the languages
        for language in languages:

            # Skip the verdict's language
            if language == row.language:
                continue

            # Translate the verdict into the given language
            raw_verdict = translator(row.raw_verdict, target_lang=language)

            # Create a new row with identical information, except the raw
            # verdict is replaced by the translated one, and the language is
            # replaced by the language
            for col_name, value in row.items():
                if col_name == 'raw_verdict':
                    annotations_dict[col_name].append(raw_verdict)
                elif col_name == 'language':
                    annotations_dict[col_name].append(language)
                else:
                    annotations_dict[col_name].append(value)

    # Convert the new data into dataframes
    new_annotations = pd.DataFrame(annotations_dict)

    # Save the dataframes to disk
    new_annotations.to_csv(new_anns_path, index=False)

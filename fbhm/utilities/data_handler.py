import matplotlib
import pandas as pd
from pathlib import Path

class DataHandler:
    def __init__(self, dfp):
        super().__init__()
        self.dfp = dfp # assumes dfp to be a Path object
        self.da = {}
    
    def load_all_data(self):
        """ Assumes jsonl file inside the data directory
            Input: DataHandler Class object, No parameters
            Output: (Pandas data frame) data frame containing folowing cols:
                    'id', 'img', 'text'.
                    The data frame contains all the unique
                    entries in all the .jsonl files irrespective of train and
                    dev and test
        """
        # Find all '.jsonl' files in data directory
        jsonl_files = sorted(Path(self.dfp).rglob('**/*.jsonl'))
        if jsonl_files is None:
            print("Data directory contains no '.jsonl' formatted files")
            raise
        
        # concatenate all data frames read from all json line files in data 
        data = pd.concat([pd.read_json(f, lines=True) for f in jsonl_files])

        # remove all duplicate entries and reindex the dataset
        return data.drop_duplicates().reset_index(drop=True)

    def compute_data_analytics(self, df):
        # Class and Shape analysis
        pd.set_option("display.precision", 2)
        self.da['SHAPE'] = df.shape
        self.da['CA'] = df['label'].value_counts(normalize=True)

        # Text analysis
        text_df = df['text'].apply(lambda s: len(s))
        fig = text_df.hist(bins='auto', alpha=0.5, ec='black').get_figure()
        fig.savefig(self.dfp/'text_len.png')
        self.da['TA'] = text_df.describe()
        return None

    # Train/Val/Test split functions
    def load_given_data(self):
        """
        Assumes file names inside data directory to be: 
            train.jsonl, dev_unseen.jsonl, test_unseen.jsonl
        returns pandad df of already provided splits for train and test images
        """
        # Set file paths
        train_path = self.dfp / 'train.jsonl'
        val_path = self.dfp / 'dev_unseen.jsonl'
        test_path = self.dfp / 'test_unseen.jsonl'

        # load into panda data frames
        data_tr = pd.read_json(train_path, lines=True)
        data_v = pd.read_json(val_path, lines=True)
        data_t = pd.read_json(test_path, lines=True)
        
        return data_tr, data_v, data_t 
    
    def smart_select(self):
        # TODO
        return None
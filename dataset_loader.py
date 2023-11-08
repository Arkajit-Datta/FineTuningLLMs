from dataclasses import dataclass
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, dataset_name: str, tokenizer):
        self.dataset_name = dataset_name
        self.dataset_loaded = False
        self.tokenizer = tokenizer
        self._load_dataset()

    def _load_dataset(self):
        # Load training split (you can process it here)
        self.train_dataset = load_dataset(self.dataset_name)
        self.train_dataset = self.train_dataset.map(self.format_example)
        self.train_dataset = self.train_dataset.remove_columns(['question', 'answer'])
        
        # Set the dataset flag
        self.dataset_loaded = True

    def get_dataset(self):
        assert self.dataset_loaded, \
            "Dataset not loaded. Please run load_dataset() first."
        self.info()
        return self.train_dataset

    def format_example(self, example):
        return {"text": f"<|system|>\n{example['system']}</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['output']}"}
    
    def info(self):
        '''
        Print the dataset info
        '''
        total_tokens = 0
        max_tokens = 0
        for i in range(len(self.train_dataset)):
            tokens = len(self.tokenizer.tokenize(self.train_dataset[i]['text']))
            if tokens > max_tokens:
                max_tokens = tokens
            total_tokens += tokens

        print(f"Total number of tokens in dataset: {total_tokens}")
        print(f"Average number of tokens: {total_tokens/len(self.train_dataset)}")
        print(f"Max. number of tokens: {max_tokens}")
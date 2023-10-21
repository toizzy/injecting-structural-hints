"""
This is the script that makes data into a huggingface dataset!

You can call 
    datasets.load_dataset("path/to/this/script/on/your/machine.py", name)
where name is one of the strings in the class Synthetic (eg. flat-parens_vocabsize-500_deplength-en)
And access any of the synthetic corpora as you would any other dataset. 

You can add new entries to the BUILDER_CONFIGS list following the pattern

Instructions about what this type of script is are here https://huggingface.co/docs/datasets/dataset_script
"""

import os
import datasets

_DESCRIPTION = """\
    Parentheses datasets for testing synthetic languages things!
"""
# TODO Change the directory to be absolute, I think it can cause some problems 
# otherwise. This directory should point to the data directory in this dir, but the 
# absolute path from your machine
_DATA_DIR = "synthetic_corpora/data"

class SyntheticConfig(datasets.BuilderConfig):

    def __init__(self, data_dir, **kwargs):
        """BuilderConfig for IzParens

        Args:
          data_dir: `string`, directory of the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(SyntheticConfig, self).__init__(
            **kwargs,
        )
        self.data_dir = data_dir

class Synthetic(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SyntheticConfig(
            name="flat-parens_deplength-en",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_deplength-en"),
            description="Flat parentheses, with empirical english deplengths",
        ),
        SyntheticConfig(
            name="flat-parens_vocab500_deplength-en",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocabsize-500_deplength-en"),
            description="Flat parentheses, with empirical english deplengths",
        ),
        SyntheticConfig(
            name="flat-parens_vocab500-zipf_deplength-en",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocabsize-500_vocab-zipf-simple_deplength-en"),
            description="Flat parentheses, with empirical english deplengths",
        ),
        SyntheticConfig(
            name="flat-parens_vocab30K-zipf_deplength-en",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocabsize-30000_vocab-zipf-simple_deplength-en"),
            description="Flat parentheses, with empirical english deplengths",
        ),
        SyntheticConfig(
            name="flat-parens_vocab-zipf-simple_deplength-en",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocab-zipf-simple_deplength-en"),
            description="Flat parentheses, with empirical english deplengths",
        ),
        SyntheticConfig(
            name="flat-parens_vocab500-zipf_deplength-long-uniform",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocab500-zipf_deplength-long-uniform"),
            description="Flat parens with long uniform deplengths",
        ),
        SyntheticConfig(
            name="flat-parens_vocab500-zipf_deplength-nesting-nolimit",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocab500-zipf_deplength-nesting-nolimit"),
            description="Flat parens with the dependency length of a dyck language",
        ),
        SyntheticConfig(
            name="flat-parens_vocab500-uniform_deplength-nesting-nolimit",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocab500-uniform_deplength-nesting-nolimit"),
            description="Flat parens with the dependency  length of a dyck language",
        ),
        SyntheticConfig(
            name="flat-parens_vocab500-zipf_deplength-nesting-nolimit_paired",
            data_dir = os.path.join(_DATA_DIR, "flat-parens_vocab500-zipf_deplength-nesting-nolimit_paired"),
            description="Flat parens with long uniform deplengths",
        ),
        SyntheticConfig(
            name="simple_repetition_vocab-zipf-simple",
            data_dir = os.path.join(_DATA_DIR, "simple_repetition_vocab-zipf-simple"),
            description="Repeated ints in sets of 10",
        ),
        SyntheticConfig(
            name="nested-parens0.49_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "nested-parens0.49_vocabsize-500_vocab-zipf-simple"),
            description="Nested Dyck language",
        ),
        SyntheticConfig(
            name="nested-parens0.49_vocab500-zipf_paired",
            data_dir = os.path.join(_DATA_DIR, "nested-parens0.49_vocab500-zipf_paired"),
            description="Nested Dyck language",
        ),
        SyntheticConfig(
            name="nested-parens0.49_vocab500-uniform",
            data_dir = os.path.join(_DATA_DIR, "nested-parens0.49_vocab500-uniform"),
            description="Nested Dyck language",
        ),
        SyntheticConfig(
            name="simple_repetition10_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "simple_repetition10_vocab500-zipf"),
            description="Repeated ints in sets of 10",
        ),
        SyntheticConfig(
            name="mod10_repetition10_vocab-500-zipf",
            data_dir = os.path.join(_DATA_DIR, "mod10_repetition10_vocabsize-500_vocab-zipf-simple"),
            description="Repeated sets of 10 with the same remainder mod10",
        ),
        SyntheticConfig(
            name="pair_repetition10_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "pair_repetition10_vocab500-zipf"),
            description="Repeated sets of 10 with different open and close tokens",
        ),
        SyntheticConfig(
            name="mod100_repetition10_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "mod100_repetition10_vocabsize-500_vocab-zipf-simple"),
            description="Repeated sets of 10 with the same remainder mod100",
        ),
        SyntheticConfig(
            name="random_vocab5000-zipf",
            data_dir = os.path.join(_DATA_DIR, "random_vocabsize-500_vocab-zipf-simple"),
            description="Randomly selected tokens",
        ),
        SyntheticConfig(
            name="random_vocab500-uniform",
            data_dir = os.path.join(_DATA_DIR, "random_vocab500-uniform"),
            description="Randomly selected tokens",
        ),
        SyntheticConfig(
            name="sparse0.1-constant10_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "sparse0.1-constant10_vocab500-zipf"),
            description="Constant deplength of 10, sparsely put",
        ),
        SyntheticConfig(
            name="sparse0.5-constant10_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "sparse0.5-constant10_vocab500-zipf"),
            description="Constant deplength of 10, sparsely put",
        ),
        SyntheticConfig(
            name="mixed-parens0.1_vocab50K-zipf",
            data_dir = os.path.join(_DATA_DIR, "mixed-parens0.1_vocab50K-zipf"),
            description="Nested, with 10% crossed mixed in",
        ),
        SyntheticConfig(
            name="mixed-parens0.1_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "mixed-parens0.1_vocab500-zipf"),
            description="Nested, with 10% crossed mixed in",
        ),
        SyntheticConfig(
            name="mixed-parens0.01_vocab500-zipf",
            data_dir = os.path.join(_DATA_DIR, "mixed-parens0.01_vocab500-zipf"),
            description="Nested, with 1% crossed mixed in",
        )
        ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_file": os.path.join(self.config.data_dir, "train.txt"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_file": os.path.join(self.config.data_dir, "test.txt"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_file": os.path.join(self.config.data_dir, "valid.txt"), "split": "valid"},
            )
        ]

    def _generate_examples(self, data_file, split):
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                row = row.strip()
                if row:
                    yield idx, {"text": row}
                else:
                    yield idx, {"text": ""}

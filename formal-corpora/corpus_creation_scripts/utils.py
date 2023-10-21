import numpy as np

LINE_LENGTH = 512
# Corpus length is in tokens, number of lines computed in the code
CORPUS_LENGTHS = {"train": 1_000_000_000, "test": 1_000_000, "valid": 500_000}

def get_distribution(vocab_distribution, vocab_size=-1, vocab_distribution_file=""):
    if vocab_distribution == "zipf":
        assert vocab_size > 0
        word_indices = np.arange(vocab_size)
        np.random.shuffle(word_indices)
        ps = [1 / (r + 2.7) for r in range(vocab_size)]
        ps = ps / np.sum(ps)
        print(word_indices)
        print(ps)
        return word_indices, ps
    elif vocab_distribution == "uniform":
        assert vocab_size > 0
        word_indices = np.arange(vocab_size)
        ps = np.ones(vocab_size)
        uniform_probability = 1.0 / vocab_size
        ps = ps * uniform_probability
        return word_indices, ps
    elif vocab_distribution == "matched":
        assert vocab_size < 0, "Can't provide a vocab size if we're matching distribution"
        distribution = json.load(open(vocab_distribution_file))
        word_indices, ps = np.zeros(len(distribution)), np.zeros(len(distribution))
        for i, (id, count) in enumerate(distribution.items()):
            word_indices[i] = id
            ps[i] = count
        ps = ps / sum(ps)
        return word_indices, ps

    #if vocab_distribution:
    #    vocab = pickle.load(open(vocab_distribution, "rb"))
    #    most_common = vocab.most_common(vocab_size - 1)
    #    frequencies = [count for word, count in most_common]
    #    # Make an unk token, that has the probability of all other words
    #    unk_frequency = sum(vocab.values()) - sum(frequencies)
    #    frequencies.append(unk_frequency)
    #    frequencies = np.array(frequencies)
    #    np.random.shuffle(frequencies)
    #    ps = frequencies / sum(frequencies)
    #    word_indices = np.arange(len(ps))
    #    return word_indices, ps

import torch
#--- Constants ---
MAX_SEQ_LENGTH = 10
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SOS_TOKEN = 36
EOS_TOKEN = 37
PAD_TOKEN = len(CHARACTERS) + 2
NUM_CLASSES = len(CHARACTERS) + 3  # SOS, EOS, PAD
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#--- Utility functions ---
def index_to_char(indices, include_special_tokens=False):
    result = []
    for i in indices:
        i = i.item() if torch.is_tensor(i) else i
        if i == SOS_TOKEN:
            if include_special_tokens: result.append('[SOS]')
        elif i == EOS_TOKEN:
            if include_special_tokens: result.append('[EOS]')
            break
        elif 0 <= i < len(CHARACTERS):
            result.append(CHARACTERS[i])
        else:
            if include_special_tokens or i not in [SOS_TOKEN, EOS_TOKEN]:
                result.append(f'[UNK_{i}]')
    return ''.join(result)

def char_to_indices(text):
    indices = [SOS_TOKEN]
    for c in text:
        if c in CHARACTERS:
            indices.append(CHARACTERS.index(c))
        else:
            indices.append(0)
    indices.append(EOS_TOKEN)
    return torch.tensor(indices, dtype=torch.long)

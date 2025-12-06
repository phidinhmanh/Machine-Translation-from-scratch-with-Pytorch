import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


dataset = load_dataset("opus100", "en-vi")
model_file = "models/spm.model"
sp = spm.SentencePieceProcessor()
sp.load(model_file) # type: ignore

unk_id = sp.unk_id()    # Unknown token
bos_id = sp.bos_id()    # Beginning of Sentence
eos_id = sp.eos_id()    # End of Sentence


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self,sp, dataset, type = 'train', max_len = 128):
        self.dataset = dataset[type]
        self.sp = sp
        self.max_len = max_len


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        en_text = self.dataset[idx]['translation']['en']
        vi_text = self.dataset[idx]['translation']['vi']
        en_ids = [bos_id] + self.sp.encode_as_ids(en_text) + [eos_id]
        vi_ids = [bos_id] +self.sp.encode_as_ids(vi_text) + [eos_id]
        return {
            'src_ids': torch.tensor(en_ids),
            'tgt_ids': torch.tensor(vi_ids)
        }

class Collectfn:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    
    def __call__(self, batch):
        src_batch = [item['src_ids'] for item in batch]
        tgt_batch = [item['tgt_ids'] for item in batch]

        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_id)

        src_mask = (src_padded != self.pad_id).long()
        tgt_mask = (tgt_padded != self.pad_id).long()

        return {
            'src_ids': src_padded,
            'tgt_ids': tgt_padded,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask
        }


# (tensor([ 96,  89,  30,  16, 711, 112,   3]), tensor([ 121,   22,   95, 3706,  468,    3]))

def getdata_loader(datasets, batch_size=32, max_len = 128, shuffle = True, type='train'):
    collate_fn = Collectfn(unk_id)
    dataset  = TranslationDataset(sp, datasets, type=type, max_len=max_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

train_loader = getdata_loader(dataset, type='train')
val_loader = getdata_loader(dataset, type='validation')
test_loader = getdata_loader(dataset, type='test')

sample = next(iter(train_loader))
print(sample['src_ids'].shape)
print(sample['tgt_ids'].shape)
print(sample['src_mask'].shape)
print(sample['tgt_mask'].shape)




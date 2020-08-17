from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers

vocab_file = '../xlm-bpe-vocab.json'
merges_file = '../xlm-bpe-merges.txt'

specials = ['<pad>', '<unk>', '<s>', '</s>']

tokenizer = Tokenizer(models.BPE(vocab_file, merges_file))
tokenizer.post_processor = processors.BertProcessing(("</s>",3),("<s>", 2))
tokenizer.decoder = decoders.BPEDecoder()
tokenizer.add_special_tokens(specials)

max_len = 15
tokenizer.enable_padding(max_length=max_len)
tokenizer.enable_truncation(max_length=max_len)

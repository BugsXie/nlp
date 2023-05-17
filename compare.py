def build_dataset(config):
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, config.class_ls_path)
        with open(config.vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        config.vocab_len = len(vocab)
        config.class_ls = [x.strip() for x in open(config.class_ls_path, 'r', encoding="utf-8").readlines()]
        print(f'\nVocab size: {len(vocab)}')

    train_data = Dataset(config.train_path, config, vocab)
    dev_data = Dataset(config.dev_path, config, vocab)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    dev_loader = DataLoader(dataset=dev_data, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

    if os.path.exists(config.test_path):
        test_data = Dataset(config.test_path, config, vocab)
        test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers)

    else:
        test_loader = dev_loader

    config.embedding_pretrained = torch.tensor(extract_vocab_tensor(config))
    return train_loader, dev_loader, test_loader

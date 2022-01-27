def run_training(id, model):
    training_summary = model.train()
    return id, training_summary

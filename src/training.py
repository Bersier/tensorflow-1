def train(model, dataset, val_dataset):
    model.fit(dataset, epochs=10, steps_per_epoch=30,
              validation_data=val_dataset,
              validation_steps=3)

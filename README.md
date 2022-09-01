# pyMachineLearningFramework

### Abstract

This is Python based Machine Learning Framework.

The purpose is to reduce the repeated part of codes in ML project

Suitable for simple and light-weight ML project

Currently support for only pytorch based deep learnig

I'm planning to add some other libraries like tensorflow


### Concepts

There's still a log of updates to do,

but I will leave some example codes for the near future.

```python
    model = Model( ... )
    model.train( ... )
    model.infer( ... )
```

The code above shows the basic class 'Model' in this framework.

It contains everything needed for some model to work out.

```python
    import torch.nn as nn

    class myNet (nn.Module) : ...
    
    my_neural_network = myNet()
    criterion = nn.MSELoss()
    with torch.no_grad():
        # train
        pred = my_neural_network(x)
        loss = criterion(pred, y)
        ...
```

For example, pytorch requires your own module to train, loss function, and some more things (like scheduler) if you need.

And Training and Validation processing is not that different for most cases.

So the model encapsulates those components into one instance and formalize training and validation process.

With the class Model, you can simply train and infer your own model. (pytorch model for now.)

But Model does not provide any data-related features. I will explain the reason later.


The next class is 'Trainer'

```python
    trainer = Trainer(model, ...)
    trainer.train( ... )
```

The code seems not much different from the previous example.

Actually if you work with only single 'model', Trainer is meaningless.

As I said before, the model class is not just a bunch of codes describing your algorithm.

It also contains several factors that affect your model.

So the class Model in this framework, refers not just 'model' but a whole group with important components except data itself.

That means you will need multiple Models inevitably for optimization or tuning.

And that's when Trainer comes out.

Trainer can handle several Models (even if the 'model' itself is same) by multiprocessing.

```python
    trainer = TorchTrainer()
    # Add models and dataloaders to trainer
    trainer.register_model([model1, model2, ...])
    trainer.add_loader([dataloader1, dataloader2, ...])
    # Link with model1 to dataloader1, and model2 to dataloader2
    trainer.link(model_index=[0, 1], loader_index=[0, 1])
    # Train model1 and model2
    trainer.train([0, 1])
```

Above example shows how it's done.

With Model and Trainer, you can easily train ML models with different hyperparameters at the same time.

### Further Plans

Current objective is to fully support pytorch.

And I'm working on supporting pytorch RPC and DDP to feature multi-GPU and remote machine learning.

Next objective is to expand the libraries to support like tensorflow.

I'm also thinking of non-neural network libraries such as SVM or RandomForest.

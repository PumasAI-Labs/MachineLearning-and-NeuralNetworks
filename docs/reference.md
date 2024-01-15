---
title: Reference Sheets for Pumas-AI Introduction to Machine Learning and Neural Networks
---

[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

## Key Points

- DeepPumas augments Pumas models with neural networks
- This workshop focuses both on DeepPumas and on machine learning concepts
- Fundamental concepts of machine learning and neural networks
    - supervised learning
    - empirical risk minimization 
    - multilayer perceptron 
    - bias-variance tradeoff
    - training, underfitting, overfitting
    - generalization
    - regularization
    - model selection
    - hyperparameter optimization
- DeepPumas basic functionalities to work with neural networks
    - `preprocess`
    - `MLPDomain`
    - `fit`
    - `optim_options`
    - `hyperopt`

## Summary of Basic Commands

| Action      | Command       | Observations          |
| ----------- | ------------- | --------------------- |
| Get a supervised machine learning dataset ready for further use with DeepPumas | `preprocess` | Expects data as matrices `X`, `Y`, with samples stored as columns, and returns a `FitTarget` for further use with `fit` and `hyperopt` |
| Construct a multilayer perceptron | `MLPDomain` | The constructor accepts parameters to specify the number of layers, the number of units in each layer, the activation functions, and the type of regularization |
| Fit a multilayer perceptron to a preprocessed supervised machine learning dataset | `fit` | It can also fit other machine learning models |
| Pass options to the optimizer | `optim_options` | A `NamedTuple` of options to be passed on to the optimizer. Here, we experiment with the number of iterations |
| Automate fitting and hyperparameter tuning | `hyperopt` | Optimize the parameters and hyperparameters of a machine learning model, in particular, of a multilayer perceptron |

<!-- TODO

## Glossary

`term1`

: Definition of the term one above.

`term2`

: Definition of the term two above. -->

## Get in touch

If you have any suggestions or want to get in touch with our education team,
please send an email to <training@pumas.ai>.

## License

This content is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

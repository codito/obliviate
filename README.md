# Obliviate

A collection of algorithms to model memory and retention of facts.

[![Build Status](https://github.com/spekt/testlogger/workflows/.NET/badge.svg)](https://github.com/spekt/testlogger/actions?query=workflow%3A.NET)
[![NuGet](https://img.shields.io/nuget/v/Obliviate.svg)](https://www.nuget.org/packages/Obliviate/)

<!--
[![NuGet Downloads](https://img.shields.io/nuget/dt/Obliviate)](https://www.nuget.org/packages/Obliviate/)
-->

## Usage

Install the nuget package in your project with `dotnet add package obliviate`.

### Ebisu

Ebisu provides a simple model that must be attached with each _fact_ the user is
trying to memorise. See the notes on [EbisuModel][] on choosing the parameters.

A learning/quizzing app will need to store the model, schedule reviews and keep
it fresh with observations from each review session. Ebisu provides two primary
APIs for these tasks. First, [PredictRecall][] attempts to find recall
probability of the existing model at a given time. E.g. _will I remember this
fact after X time units from the last review?_ Second, assume we reviewed the
fact `n` times with `k` successful reviews after `t` time units from last
review. [UpdateRecall][] updates the previous model with these additional
observations.

Ebisu provides fantastic documentation [here][ebisu]. We highly recommend a read
if you're planning to use the algorithm.

[ebisumodel]: https://github.com/codito/obliviate/blob/master/src/Obliviate/Ebisu/EbisuModel.cs
[predictrecall]: https://github.com/codito/obliviate/blob/54e74e55fd27bd4681c94bef8c60acd5f90aaabd/src/Obliviate/Ebisu/EbisuModelExtensions.cs#L29
[updaterecall]: https://github.com/codito/obliviate/blob/54e74e55fd27bd4681c94bef8c60acd5f90aaabd/src/Obliviate/Ebisu/EbisuModelExtensions.cs#L68
[ebisu]: https://fasiha.github.io/ebisu/

## Algorithms

- [x] Ebisu: https://fasiha.github.io/ebisu/ v2.0.0 (Public domain)
  - [ ] Ebisu v2.1.0 support with soft binary quizzes and half life rescale
- [ ] Memorize: https://github.com/Networks-Learning/memorize (MIT)
- [ ] Duolingo Halflife: https://github.com/duolingo/halflife-regression (MIT)
- [ ] SM-2 and related family of algorithms

We plan to support these algorithms along with benchmarks in future. Contributions
and suggestions are most welcome!

## License

MIT

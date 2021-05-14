// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Ebisu
{
    /// <summary>
    /// Ebisu model encodes the probability of recall of a fact using Beta distribution.
    /// <c>Alpha</c> and <c>Beta</c> define the shape parameters of the distribution and
    /// <c>Time</c> represents the recall time.
    /// </summary>
    /// <remarks>
    /// The application needs to attach a model to each fact that is reviewed.
    /// How to pick the parameters?
    /// Set <c>Time</c> to the best guess of fact's half life. E.g. Memrise keeps this
    /// at 4hrs, Anki uses 1 day and Ebisu author recommends 15mins.
    /// For <c>Time</c> to represent half life, <c>Alpha = Beta</c>. Choose a value more
    /// than 1. Author recommends 3 as a good default as it aggressively changes the half life
    /// based on results. Higher value of <c>Alpha</c> and <c>Beta</c> implies lesser change in
    /// the recall duration (i.e. more stability).
    /// </remarks>
    public record EbisuModel(double Time, double Alpha, double Beta);
}
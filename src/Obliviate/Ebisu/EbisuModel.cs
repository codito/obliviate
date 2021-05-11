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
    /// </remarks>
    public record EbisuModel(double Time, double Alpha, double Beta);
}
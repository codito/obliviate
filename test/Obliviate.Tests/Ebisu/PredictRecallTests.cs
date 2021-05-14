// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Tests.Ebisu
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Obliviate.Ebisu;

    [TestClass]
    public class PredictRecallTests
    {
        private const double Tolerance = 5e-8;

        [TestMethod]
        [DataRow(1.0, 1.0, 1.0, 0.0, 0.0)]
        [DataRow(2.0, 2.0, 2.0, 2.0, -0.69314718)]
        public void ShouldReturnProbabilityOfRecallAfterDuration(
            double alpha,
            double beta,
            double time,
            double duration,
            double expectedRecall)
        {
            var prior = new EbisuModel(time, alpha, beta);

            var recall = prior.PredictRecall(duration);

            Assert.AreEqual(expectedRecall, recall, Tolerance);
        }

        [TestMethod]
        [DataRow(1.0, 1.0, 1.0, 0.0, 1.0)]
        [DataRow(2.0, 2.0, 2.0, 2.0, 0.5)]
        public void ShouldReturnExactProbabilityOfRecallAfterDuration(
            double alpha,
            double beta,
            double time,
            double duration,
            double expectedRecall)
        {
            var prior = new EbisuModel(time, alpha, beta);

            var recall = prior.PredictRecall(duration, exact: true);

            Assert.AreEqual(expectedRecall, recall, Tolerance);
        }
    }
}
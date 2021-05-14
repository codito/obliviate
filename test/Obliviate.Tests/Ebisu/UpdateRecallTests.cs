// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Tests.Ebisu
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Obliviate.Ebisu;

    [TestClass]
    public class UpdateRecallTests
    {
        private const double Tolerance = 5e-8;

        [TestMethod]
        [DataRow(new[] { 2.0, 2.0, 2.0 }, 1, 1, 2.0, new[] { 3.0, 2.0, 0.0 })] // success recalls
        [DataRow(new[] { 2.0, 2.0, 2.0 }, 0, 1, 2.0, new[] { 2.0, 3.0, 0.0 })] // fail recalls
        public void ShouldReturnModelWithUpdatedParameters(
            double[] priorParams,
            int successes,
            int total,
            double duration,
            double[] expectedParams)
        {
            var prior = new EbisuModel(priorParams[2], priorParams[0], priorParams[1]);
            var expected = new EbisuModel(expectedParams[2], expectedParams[0], expectedParams[1]);

            var actual = prior.UpdateRecall(successes, total, duration);

            Assert.AreEqual(expected.Alpha, actual.Alpha, Tolerance);
        }
    }
}
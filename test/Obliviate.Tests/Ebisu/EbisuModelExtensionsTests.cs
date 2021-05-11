// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Tests.Ebisu
{
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Obliviate.Ebisu;

    [TestClass]
    public class EbisuModelExtensionsTests
    {
        private const double Epsilon = 1e-12;

        [TestMethod]
        public void UpdateRecall()
        {
            var m = new EbisuModel(2, 2, 2);
            var success = m.UpdateRecall(1, 1, 2.0);
            var failure = m.UpdateRecall(0, 1, 2.0);

            Assert.AreEqual(3.0, success.Alpha, Epsilon, "success/alpha");
            Assert.AreEqual(2.0, success.Beta, Epsilon, "success/beta");
            Assert.AreEqual(2.0, failure.Alpha, Epsilon, "failure/alpha");
            Assert.AreEqual(3.0, failure.Beta, Epsilon, "failure/beta");
        }
    }
}
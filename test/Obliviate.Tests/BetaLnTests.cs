// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Obliviate.Ebisu;

    [TestClass]
    public class BetaLnTests
    {
        /// <summary>
        /// Test for behavioral difference between the two dependencies: Infer.net
        /// and Math numeric on the implementation of BetaLn function.
        /// </summary>
        [TestMethod]
        public void BetaLnFunctionForZeroSuccesses()
        {
            var delta = 0.05;
            var prior = new EbisuModel(1.0, 34.4, 3.4);

            var updatedModel = prior.UpdateRecall(0, 5, 0.1);

            Assert.AreEqual(3.0652051705190964, updatedModel.Time, delta);
            Assert.AreEqual(8.706432410647471, updatedModel.Beta, delta);
            Assert.AreEqual(8.760308130181903, updatedModel.Alpha, delta);

#if NONE
            double timeNow = 0.1;
            double timeBack = prior.Time;
            int successes = 0;
            int total = 5;
            double alpha = prior.Alpha;
            double beta = prior.Beta;
            double t = prior.Time;
            double dt = timeNow / t;
            double et = timeBack / timeNow;
            var failures = total - successes;

            var binomlns = Enumerable.Range(0, failures + 1)
                .Select(i => BinomialLn(failures, i)).ToArray();
            var logs1 =
                Enumerable.Range(0, 3)
                    .Select(m =>
                    {
                        var a =
                            Enumerable.Range(0, failures + 1)
                                .Select(i => binomlns[i] + SpecialFunctions.BetaLn(
                                    beta,
                                    alpha + (dt * (successes + i)) + (m * dt * et)))
                                .ToList();
                        var b = Enumerable.Range(0, failures + 1)
                            .Select(i => Math.Pow(-1.0, i))
                            .ToList();
                        return LogSumExp(a, b)[0];
                    })
                    .ToArray();
            var logs2 =
                Enumerable.Range(0, 3)
                    .Select(m =>
                    {
                        var a =
                            Enumerable.Range(0, failures + 1)
                                .Select(i => binomlns[i] + Beta.BetaLn(
                                    beta,
                                    alpha + (dt * (successes + i)) + (m * dt * et)))
                                .ToList();
                        var b = Enumerable.Range(0, failures + 1)
                            .Select(i => Math.Pow(-1.0, i))
                            .ToList();
                        return LogSumExp(a, b)[0];
                    })
                    .ToArray();

            for (int i = 0; i < logs1.Length; i++)
            {
                Assert.AreEqual(logs1[i], logs2[i], 1e-3);
            }
#endif
        }

        private static double BinomialLn(int n, int k)
        {
            return -SpecialFunctions.BetaLn(1.0 + n - k, 1.0 + k) - Math.Log(n + 1.0);
        }

        private static double[] LogSumExp(List<double> a, List<double> b)
        {
            double amax = a.Max();
            double sum = Enumerable.Range(0, a.Count)
                .Select(i => Math.Exp(a[i] - amax) * (i < b.Count ? b[i] : 1.0))
                .Sum();
            double sign = Math.Sign(sum);
            sum *= sign;
            double abs = Math.Log(sum) + amax;
            double[] ret = { abs, sign };
            return ret;
        }
    }
}
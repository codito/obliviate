// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Ebisu
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.Optimization;
    using Microsoft.ML.Probabilistic.Distributions;

    public static class EbisuModelExtensions
    {
        /// <summary>
        /// Estimate the recall probability of an existing model given the time units
        /// elapsed since last review.
        /// </summary>
        /// <param name="prior">Existing ebisu model.</param>
        /// <param name="timeNow">Time elapsed since last review.</param>
        /// <param name="exact">Return log probabilities if false (default).</param>
        /// <returns>Probability of recall. 0 represents fail, and 1 for pass.</returns>
        public static double PredictRecall(
            this EbisuModel prior,
            double timeNow,
            bool exact = false)
        {
            double alpha = prior.Alpha;
            double beta = prior.Beta;
            double dt = timeNow / prior.Time;
            double ret = BetaLn(alpha + dt, beta) -
                         BetaLn(alpha, beta);
            return exact ? Math.Exp(ret) : ret;
        }

        /// <summary>
        /// Update the parameters of a prior model with new observations and return
        /// an updated model with posterior distribution of recall probability at
        /// <paramref name="timeNow"/> time units after review.
        /// </summary>
        /// <param name="prior">Existing model representing the beta distribution for a fact.</param>
        /// <param name="successes">Number of successful reviews for the fact.</param>
        /// <param name="total">Number of total reviews for the fact.</param>
        /// <param name="timeNow">Elapsed time units since last review was recorded.</param>
        /// <returns>Updated model for the fact.</returns>
        /// <remarks>By default, this method will rebalance the returned model to represent
        /// recall probability distribution after half life time units since last review.
        /// See <c>UpdateRecall</c> overload to modify this behavior.</remarks>
        public static EbisuModel UpdateRecall(
            this EbisuModel prior,
            int successes,
            int total,
            double timeNow)
        {
            return prior.UpdateRecall(
                successes,
                total,
                timeNow,
                true,
                prior.Time);
        }

        /// <summary>
        /// Update the parameters of a prior model with new observations and return
        /// an updated model with posterior distribution of recall probability at
        /// <paramref name="timeBack"/> time units after review.
        /// </summary>
        /// <param name="prior">Existing model representing the beta distribution for a fact.</param>
        /// <param name="successes">Number of successful reviews for the fact.</param>
        /// <param name="total">Number of total reviews for the fact.</param>
        /// <param name="timeNow">Elapsed time units since last review was recorded.</param>
        /// <param name="rebalance">If true, the updated model is computed with <paramref name="timeBack"/> set to half life.</param>
        /// <param name="timeBack">Time stamp for calculating recall in the updated model.</param>
        /// <returns>Updated model for the fact.</returns>
        public static EbisuModel UpdateRecall(
            this EbisuModel prior,
            int successes,
            int total,
            double timeNow,
            bool rebalance,
            double timeBack)
        {
            double alpha = prior.Alpha;
            double beta = prior.Beta;
            double t = prior.Time;
            double dt = timeNow / t;
            double et = timeBack / timeNow;
            var failures = total - successes;

            var binomlns = Enumerable.Range(0, failures + 1)
                .Select(i => BinomialLn(failures, i)).ToArray();
            var logs =
                Enumerable.Range(0, 3)
                    .Select(m =>
                    {
                        var a =
                            Enumerable.Range(0, failures + 1)
                                .Select(i => binomlns[i] + BetaLn(
                                    beta,
                                    alpha + (dt * (successes + i)) + (m * dt * et)))
                                .ToList();
                        var b = Enumerable.Range(0, failures + 1)
                            .Select(i => Math.Pow(-1.0, i))
                            .ToList();
                        return LogSumExp(a, b)[0];
                    })
                    .ToArray();

            double logDenominator = logs[0];
            double logMeanNum = logs[1];
            double logM2Num = logs[2];

            double mean = Math.Exp(logMeanNum - logDenominator);
            double m2 = Math.Exp(logM2Num - logDenominator);
            double meanSq = Math.Exp(2 * (logMeanNum - logDenominator));
            double sig2 = m2 - meanSq;

            if (mean <= 0)
            {
                throw new("invalid mean found");
            }

            if (m2 <= 0)
            {
                throw new("invalid second moment found");
            }

            if (sig2 <= 0)
            {
                throw new(
                    $"invalid variance found a={alpha}, b={beta}, t={t}, k={successes}, n={total}, tnow={timeNow}, mean={mean}, m2={m2}, sig2={sig2}");
            }

            List<double> newAlphaBeta = MeanVarToBeta(mean, sig2);
            EbisuModel proposed = new EbisuModel(timeBack, newAlphaBeta[0], newAlphaBeta[1]);

            return rebalance ? prior.Rebalance(successes, total, timeNow, proposed) : proposed;
        }

        private static double BinomialLn(int n, int k)
        {
            return -BetaLn(1.0 + n - k, 1.0 + k) - Math.Log(n + 1.0);
        }

        /// <summary>
        /// Stably evaluate the log of the sum of the exponentials of inputs.
        ///
        /// The basic idea is, you have a bunch of numbers in the log domain, e.g., the
        /// results of `logGamma`. Then you want to sum them, but you cannot sum in the
        /// log domain: you have to apply `exp` first before summing. But if you have
        /// very big values, `exp` might overflow (this is probably why you started out
        /// with the log domain in the first place!). This function lets you do the sum
        /// more stably, and returns the result of the sum in the log domain.
        ///
        /// See
        /// https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        ///
        /// Analogous to `log(sum(b .* exp(a)))` (in Python/Julia notation). `b`'s
        /// values default to 1.0 if `b` is not as long as `a`.
        ///
        /// Because the elements of `b` can be negative, to effect subtraction, the
        /// result might be negative. Therefore, two numbers are returned: the absolute
        /// value of the result, and its sign.
        /// </summary>
        /// <param name="a">Logs of the values to be summed.</param>
        /// <param name="b">Scalars to be applied element-wise to <c>exp(a)</c>.</param>
        /// <returns>2-array containing result's absolute value and its sign (1 or -1).</returns>
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

        /// <summary>
        /// Convert the mean and variance of a Beta distribution to its parameters.
        ///
        /// See https://en.wikipedia.org/wiki/Beta_distribution#Two_parameters.
        /// </summary>
        /// <param name="mean">x̄ in the Wikipedia reference above.</param>
        /// <param name="v">v̄ in the Wikipedia reference above.</param>
        /// <returns>a 2-element list containing `alpha` and `beta`.</returns>
        private static List<double> MeanVarToBeta(double mean, double v)
        {
            double tmp = (mean * (1 - mean) / v) - 1;
            double alpha = mean * tmp;
            double beta = (1 - mean) * tmp;
            return new() { alpha, beta };
        }

        private static EbisuModel Rebalance(
            this EbisuModel prior,
            int successes,
            int total,
            double timeNow,
            EbisuModel proposed)
        {
            double newAlpha = proposed.Alpha;
            double newBeta = proposed.Beta;
            if (newAlpha > 2 * newBeta || newBeta > 2 * newAlpha)
            {
                // Compute the elapsed time for this model to reach half its recall
                // probability i.e. half life
                double roughHalflife = ModelToPercentileDecay(proposed, 0.5, true, 1e-4);
                return prior.UpdateRecall(successes, total, timeNow, false, roughHalflife);
            }

            return proposed;
        }

        private static double ModelToPercentileDecay(
            this EbisuModel model,
            double percentile,
            bool coarse,
            double tolerance)
        {
            if (percentile < 0 || percentile > 1)
            {
                throw new("percentiles must be between (0, 1) exclusive");
            }

            double alpha = model.Alpha;
            double beta = model.Beta;
            double t0 = model.Time;

            double logBab = BetaLn(alpha, beta);
            double logPercentile = Math.Log(percentile);
            Func<double, double> f = lndelta =>
            {
                return (BetaLn(alpha + Math.Exp(lndelta), beta) - logBab) -
                       logPercentile;
            };

            double bracket_width = coarse ? 1.0 : 6.0;
            double blow = -bracket_width / 2.0;
            double bhigh = bracket_width / 2.0;
            double flow = f(blow);
            double fhigh = f(bhigh);
            while (flow > 0 && fhigh > 0)
            {
                // Move the bracket up.
                blow = bhigh;
                flow = fhigh;
                bhigh += bracket_width;
                fhigh = f(bhigh);
            }

            while (flow < 0 && fhigh < 0)
            {
                // Move the bracket down.
                bhigh = blow;
                fhigh = flow;
                blow -= bracket_width;
                flow = f(blow);
            }

            if (!(flow > 0 && fhigh < 0))
            {
                throw new("failed to bracket");
            }

            if (coarse)
            {
                return (Math.Exp(blow) + Math.Exp(bhigh)) / 2 * t0;
            }

            var status = GoldenSectionMinimizer.Minimum(
                ObjectiveFunction.ScalarValue(y => Math.Abs(f(y))),
                blow,
                bhigh,
                tolerance,
                10000);
            double sol = status.MinimizingPoint;
            return Math.Exp(sol) * t0;
        }

        private static double BetaLn(double z, double w)
        {
#if NONE
            return SpecialFunctions.BetaLn(z, w);
#else
            return Beta.BetaLn(z, w);
#endif
        }
    }
}
// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Ebisu
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.RootFinding;
    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// Port of Ebisu memory model.
    /// </summary>
    /// <remarks>
    /// Algorithms synced to v2.0.0 of Ebisu
    /// https://github.com/fasiha/ebisu/blob/d338cc8e45693bb17c4b9b72b9f63b78949d264c/ebisu/ebisu.py.
    /// </remarks>
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

            // Ebisu represents the events as a GB1 distribution. Expected recall
            // probability is `B(alpha + dt, beta)/B(alpha, beta)`, where `B()` is
            // the beta function. We are calculating it over the log domain.
            // So `log(a/b) = log(a) - log(b)` applies.
            // See https://en.wikipedia.org/wiki/Generalized_beta_distribution#Generalized_beta_of_first_kind_(GB1)
            // and the notes at https://fasiha.github.io/ebisu/ (Recall probability right now).
            double ret = BetaLn(alpha + dt, beta) -
                         BetaLn(alpha, beta);

            return exact ? Math.Exp(ret) : ret;
        }

        /// <summary>
        /// Update the parameters of a prior model with new observations and return
        /// an updated model with posterior distribution of recall probability at
        /// <paramref name="timeNow"/> time units after review.
        ///
        /// <paramref name="prior"/> is the given belief about remembrance of the fact. We
        /// attempt to calculate the posterior given additional data i.e. <paramref name="successes"/>
        /// indicating the successful recalls in <paramref name="total"/> review attempts in
        /// <paramref name="timeNow"/> duration since last review.
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
        ///
        /// <paramref name="prior"/> is the given belief about remembrance of the fact. We
        /// attempt to calculate the posterior given additional data i.e. <paramref name="successes"/>
        /// indicating the successful recalls in <paramref name="total"/> review attempts in
        /// <paramref name="timeNow"/> duration since last review.
        /// </summary>
        /// <param name="prior">Existing model representing the beta distribution for a fact.</param>
        /// <param name="successes">Number of successful reviews for the fact.</param>
        /// <param name="total">Number of total reviews for the fact.</param>
        /// <param name="timeNow">Elapsed time units since last review was recorded.</param>
        /// <param name="rebalance">If true, the updated model is computed with <paramref name="timeBack"/> set to half life.</param>
        /// <param name="timeBack">Time stamp for calculating recall in the updated model.</param>
        /// <returns>Updated model for the fact.</returns>
        /// <remarks>
        /// Each review of the fact can be modelled as a binomial experiment with
        /// `k` successes in `n` trials. These are represented as the `successes` and
        /// `total` variables here. If `total` is 1, this is a bernoulli experiment.
        /// Second, we're assuming the experiments (reviews) to be independent. They're not
        /// independent if the app shows a hint to the user, obviously the next review is biased.
        ///
        /// Given the `prior` recall probability and the results of new experiments, what is the
        /// `posterior` recall probability? Note we're being bayesian, and asking hard questions.
        /// </remarks>
        public static EbisuModel UpdateRecall(
            this EbisuModel prior,
            int successes,
            int total,
            double timeNow,
            bool rebalance,
            double timeBack)
        {
            if (successes < 0 || successes > total)
            {
                throw new ArgumentException(
                    "Successes must not be negative and less than Total.",
                    nameof(successes));
            }

            if (total < 1)
            {
                throw new ArgumentException(
                    "Total experiments must be one or more.",
                    nameof(total));
            }

            // See https://fasiha.github.io/ebisu/ (Updating the posterior with quiz results)
            // section for detailed derivation.
            double alpha = prior.Alpha;
            double beta = prior.Beta;
            double t = prior.Time;
            double dt = timeNow / t;
            double et = timeBack / timeNow;
            var failures = total - successes;

            // Most of the calculations are summations over the range [0, failures]
            var binomlns = Enumerable.Range(0, failures + 1)
                .Select(i => BinomialLn(failures, i)).ToArray();
            var logs = Enumerable.Range(0, 3)
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
                    return LogSumExp(a, b).Value;
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
                throw new EbisuConstraintViolationException($"Invalid mean found: a={alpha}, b={beta}, t={t}, k={successes}, n={total}, tnow={timeNow}, mean={mean}, m2={m2}, sig2={sig2}");
            }

            if (m2 <= 0)
            {
                throw new EbisuConstraintViolationException($"Invalid second moment found: a={alpha}, b={beta}, t={t}, k={successes}, n={total}, tnow={timeNow}, mean={mean}, m2={m2}, sig2={sig2}");
            }

            if (sig2 <= 0)
            {
                throw new EbisuConstraintViolationException(
                    $"Invalid variance found: a={alpha}, b={beta}, t={t}, k={successes}, n={total}, tnow={timeNow}, mean={mean}, m2={m2}, sig2={sig2}");
            }

            // Compute the Beta function from mean and variance
            // See https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
            var (newAlpha, newBeta) = MeanVarToBeta(mean, sig2);
            var proposed = new EbisuModel(timeBack, newAlpha, newBeta);

            return rebalance ? prior.Rebalance(successes, total, timeNow, proposed) : proposed;
        }

        /// <summary>
        /// Calculate the binomial coefficient over logarithmic domain.
        /// </summary>
        /// <param name="n">Total number of experiments.</param>
        /// <param name="k">Number of successful observations.</param>
        /// <returns>Log of binomial coefficient.</returns>
        private static double BinomialLn(int n, int k)
        {
            // See https://proofwiki.org/wiki/Binomial_Coefficient_expressed_using_Beta_Function
            return -BetaLn(1.0 + n - k, 1.0 + k) - Math.Log(n + 1.0);
        }

        /// <summary>
        /// Stably evaluate the log of the sum of the exponentials of inputs.
        ///
        /// The basic idea is, you have a bunch of numbers in the log domain, e.g., the
        /// results of <c>logGamma</c>. Then you want to sum them, but you cannot sum in the
        /// log domain: you have to apply <c>exp</c> first before summing. But if you have
        /// very big values, <c>exp</c> might overflow (this is probably why you started out
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
        /// <returns>Tuple containing result's absolute value and its sign (1 or -1).</returns>
        private static (double Value, int Sign) LogSumExp(List<double> a, List<double> b)
        {
            double amax = a.Max();
            double sum = Enumerable.Range(0, a.Count)
                .Select(i => Math.Exp(a[i] - amax) * (i < b.Count ? b[i] : 1.0))
                .Sum();
            int sign = Math.Sign(sum);
            sum *= sign;
            double abs = Math.Log(sum) + amax;
            return (abs, sign);
        }

        /// <summary>
        /// Convert the mean and variance of a Beta distribution to its parameters.
        ///
        /// See https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance.
        /// </summary>
        /// <param name="mean"><c>x̄</c> in the Wikipedia reference above.</param>
        /// <param name="v"><c>v̄</c> in the Wikipedia reference above.</param>
        /// <returns>Tuple containing <c>alpha</c> and <c>beta</c>.</returns>
        private static (double Alpha, double Beta) MeanVarToBeta(double mean, double v)
        {
            double tmp = (mean * (1 - mean) / v) - 1;
            double alpha = mean * tmp;
            double beta = (1 - mean) * tmp;
            return (alpha, beta);
        }

        /// <summary>
        /// Rebalance a proposed posterior model to ensure its <c>Alpha</c> and
        /// <c>Beta</c> parameters are close.
        /// Since <c>Alpha = Beta</c> implies half life, this operation keeps
        /// tries to update the shape parameters for numerical stability.
        /// </summary>
        /// <param name="prior">Existing memory model.</param>
        /// <param name="successes">Count of successful reviews.</param>
        /// <param name="total">Count of total number of reviews.</param>
        /// <param name="timeNow">Duration since last review.</param>
        /// <param name="proposed">Proposed memory model.</param>
        /// <returns>Updated model with duration nearer to the half life.</returns>
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

        /// <summary>
        /// Compute the time duration for a <see cref="EbisuModel"/> to decay to
        /// a given percentile.
        /// </summary>
        /// <param name="model">Given model for the fact.</param>
        /// <param name="percentile">Target percentile for the decay.</param>
        /// <param name="coarse">If true, use an approximation for the duration returned.</param>
        /// <param name="tolerance">Allowed tolerance for the duration.</param>
        /// <returns>Duration in time units (of provided model) for the decay to given percentile.</returns>
        private static double ModelToPercentileDecay(
            this EbisuModel model,
            double percentile,
            bool coarse,
            double tolerance)
        {
            if (percentile < 0 || percentile > 1)
            {
                throw new ArgumentException(
                    "Percentiles must be between (0, 1) exclusive",
                    nameof(percentile));
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

            double bracketWidth = coarse ? 1.0 : 6.0;
            double blow = -bracketWidth / 2.0;
            double bhigh = bracketWidth / 2.0;
            double flow = f(blow);
            double fhigh = f(bhigh);
            while (flow > 0 && fhigh > 0)
            {
                // Move the bracket up.
                blow = bhigh;
                flow = fhigh;
                bhigh += bracketWidth;
                fhigh = f(bhigh);
            }

            while (flow < 0 && fhigh < 0)
            {
                // Move the bracket down.
                bhigh = blow;
                fhigh = flow;
                blow -= bracketWidth;
                flow = f(blow);
            }

            if (!(flow > 0 && fhigh < 0))
            {
                throw new EbisuConstraintViolationException($"Failed to bracket: flow={flow}, fhigh={fhigh}");
            }

            if (coarse)
            {
                return (Math.Exp(blow) + Math.Exp(bhigh)) / 2 * t0;
            }

            // Similar to the `root_scalar` api with bracketing
            // See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html#scipy.optimize.root_scalar
            var sol = Brent.FindRoot(f, blow, bhigh);
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
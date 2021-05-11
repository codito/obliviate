// Copyright (c) Arun Mahapatra. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace Obliviate.Tests.Ebisu
{
    using System.IO;
    using System.Linq;
    using System.Text.Json;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Obliviate.Ebisu;

    /// <summary>
    /// Tests this implementation against the reference ebisu.py implementation.
    /// </summary>
    [TestClass]
    public class EbisuReferenceTests
    {
        private const double Tolerance = 5e-2; // 5e-3;

        [TestMethod]
        public void MatchReferenceImplementation()
        {
            var stream = File.OpenRead("Ebisu/test.json");
            using var testData = JsonDocument.Parse(stream);

            // Format of the test.json:
            // Array of test cases, where each test case is one of the following
            //  ["update", [a, b, t0], [k, n, t], {"post": [a, b, t]}]
            //  ["predict", [a, b, t0], [t], {"post": [mean]}]
            // test.json can be obtained from the ebisu python repository.
            foreach (var test in testData.RootElement.EnumerateArray())
            {
                var data = test.EnumerateArray().ToArray();
                var operation = data[0].GetString();
                var modelData = data[1].EnumerateArray().ToArray();
                var model = new EbisuModel(
                    modelData[2].GetDouble(),
                    modelData[0].GetDouble(),
                    modelData[1].GetDouble());

                switch (operation)
                {
                    case "update":
                        var paramData = data[2].EnumerateArray().ToArray();
                        var successes = paramData[0].GetInt32();
                        var total = paramData[1].GetInt32();
                        var time = paramData[2].GetDouble();
                        var expectedData = data[3].EnumerateObject().First()
                            .Value.EnumerateArray().ToArray();
                        var expected = new EbisuModel(
                            expectedData[2].GetDouble(),
                            expectedData[0].GetDouble(),
                            expectedData[1].GetDouble());

                        var updatedModel = model.UpdateRecall(successes, total, time);

                        Assert.AreEqual(expected.Time, updatedModel.Time, Tolerance, $"Test: {test}");
                        Assert.AreEqual(expected.Alpha, updatedModel.Alpha, Tolerance, $"Test: {test}");
                        Assert.AreEqual(expected.Beta, updatedModel.Beta, Tolerance, $"Test: {test}");
                        break;
                    case "predict":
                        time = data[2].EnumerateArray().First().GetDouble();
                        var expectedRecall = data[3].EnumerateObject().First().Value
                            .GetDouble();

                        var predictRecall = model.PredictRecall(time, true);

                        Assert.AreEqual(expectedRecall, predictRecall, Tolerance);
                        break;
                    default:
                        Assert.Fail("Reference data has invalid operation.");
                        break;
                }
            }
        }
    }
}
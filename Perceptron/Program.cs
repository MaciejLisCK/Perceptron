using System;
using System.Collections.Generic;
using System.Linq;

namespace Perceptron
{
    class Program
    {
        static readonly Random Random = new Random(1);
        static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        static double SigmoidDerivative(double x) => x * (1 - x);

        static int[][] trainingInputs = new int[][]
        {
            new [] { 0,0,1 },
            new [] { 1,1,1 },
            new [] { 1,0,1 },
            new [] { 0,1,1 },
        };
        static int[] trainingOutputs = new[] { 0, 1, 1, 0 };
        static double[] weights = new double[] { -0.16595599d, 0.44064899d, -0.99977125d };

        static void Main(string[] args)
        {
            // training network
            // iteration
            for (int i = 0; i < 1000; i++)
            {
                // training set
                for (int j = 0; j < trainingInputs.Length; j++)
                {
                    var trainingInput = trainingInputs[j];
                    var trainingOutput = trainingOutputs[j];

                    // calculate output
                    double sum = 0;
                    for (int k = 0; k < trainingInput.Length; k++)
                    {
                        sum += trainingInput[k] * weights[k];
                    }
                    var output = Sigmoid(sum);

                    // calculate error
                    var error = trainingOutput - output;

                    // calculate adjustment
                    var adjustment = error * SigmoidDerivative(output);

                    // set new weights
                    for (int k = 0; k < trainingInput.Length; k++)
                    {
                        weights[k] += trainingInput[k] * adjustment;
                    }
                }
            }

            Console.WriteLine("Network trained");

            // test network
            while (true)
            {
                Console.WriteLine("Enter 3 values followed by enter from set [0,1]");
                var input1 = Double.Parse(Console.ReadLine());
                var input2 = Double.Parse(Console.ReadLine());
                var input3 = Double.Parse(Console.ReadLine());

                var userInput = new double[] { input1, input2, input3 };

                // calculate output
                double sum = 0;
                for (int k = 0; k < userInput.Length; k++)
                {
                    sum += userInput[k] * weights[k];
                }
                var userOutput = Sigmoid(sum);

                Console.WriteLine("Output of network: " + userOutput);
            }
        }
    }
}

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

        static void Main(string[] args)
        {
            var training_inputs = new int[][]
            {
                new [] { 0,0,1 },
                new [] { 1,1,1 },
                new [] { 1,0,1 },
                new [] { 0,1,1 },
            };

            var training_outputs = new[] { 0, 1, 1, 0 };

            var synaptic_weights = new double[] { -0.16595599d, 0.44064899d, -0.99977125d }.ToList();

            // training network
            // iteration
            for (int i = 0; i < 1000; i++)
            {
                // training set
                for (int j = 0; j < training_inputs.Length; j++)
                {
                    var training_input = training_inputs[j];
                    var training_output = training_outputs[j];

                    // calculate output
                    double sum = 0;
                    for (int k = 0; k < training_input.Length; k++)
                    {
                        sum += training_input[k] * synaptic_weights[k];
                    }
                    var output = Sigmoid(sum);

                    // calculate error
                    var error = training_output - output;

                    // calculate adjustment
                    var adjustment = error * SigmoidDerivative(output);

                    // set new weights
                    for (int k = 0; k < training_input.Length; k++)
                    {
                        synaptic_weights[k] += training_input[k] * adjustment;
                    }
                }
            }

            Console.WriteLine("Network trained");

            // test network
            while (true)
            {
                Console.WriteLine("Enter 3 values followed by [enter] from set [0,1]");
                var input1 = Double.Parse(Console.ReadLine());
                var input2 = Double.Parse(Console.ReadLine());
                var input3 = Double.Parse(Console.ReadLine());

                var user_input = new double[] { input1, input2, input3 };

                // calculate output
                double sum = 0;
                for (int k = 0; k < user_input.Length; k++)
                {
                    sum += user_input[k] * synaptic_weights[k];
                }
                var user_output = Sigmoid(sum);

                Console.WriteLine("Output of network: " + user_output);
            }
        }
    }
}

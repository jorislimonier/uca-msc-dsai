using System;
using System.Linq;

namespace Test
{
    class Perceptron
    {
        static void Main()
        {
            int[,] input = new int[,] { { 0, 0 }, { 1, 1 }, { 1, 0 }, { 0, 1 } };
            int[] outputs = { 0, 1, 0, 0 };

            Random r = new Random();

            double[] weights = { r.NextDouble(), r.NextDouble(), r.NextDouble() };

            double learningRate = 0.5;
            double totalError = 1;
            int nbSteps = 0;
            int nbStepsMax = 1000;

            while (totalError > 0.2 && nbSteps < nbStepsMax)
            {
                nbSteps += 1;
                totalError = 0;

                for (int i = 0; i < 4; i++)
                {
                    int output = calculateOutput(input[i, 0], input[i, 1], weights);

                    int error = outputs[i] - output;

                    weights[0] += learningRate * error * input[i, 0];
                    weights[1] += learningRate * error * input[i, 1];
                    weights[2] += learningRate * error * 1;

                    totalError += Math.Abs(error);
                    Array.ForEach(weights, Console.WriteLine);
                    
                }
            }
            System.Console.WriteLine($"Results after {nbSteps} steps (error: {totalError}):");
            for (int i=0;i<4;i++)
                System.Console.WriteLine(calculateOutput(input[i, 0], input[i, 1], weights));
        }

        private static int calculateOutput(double input0, double input1, double[] weights)
        {
            double sum = input0 * weights[0] + input1 * weights[1] + weights[2];
            return (sum >= 0) ? 1 : 0;
        }
    }
}
using System;
using System.Collections.Generic;

namespace NerualNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public NeuronType NeuType { get; }
        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuType = type;
            Weights = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            var sum = 0.0;
            for(int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            Output = Sigmoid(sum);
            return Output;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        public void SetWeigths(params double[] weigths)
        {
            // TODO: Удалить после обучения сети
            for(int i = 0; i < weigths.Length; i++)
            {
                Weights[i] = weigths[i];
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}

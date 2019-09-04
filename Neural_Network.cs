using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetwork
{
    public class Nerual_Network
    {
        static void Main(string[] args){}
        public Topology Topology { get; }
        public List<Layer> layers { get; }

        public Nerual_Network(Topology topology)
        {
            Topology = topology;

            layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if(Topology.OutputCount == 1)
            {
                return layers.Last().Neurons[0];
            }
            else
            {
                return layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += BackPropagation(output, input);
                }
            }

            var result = error / epoch;
            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row, i];
            return array;
        }

        private double[,] Scaling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for(int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for(int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if(item < min)
                    {
                        min = item;
                    }
                    if(item > max)
                    {
                        max = item;
                    }
                }
                var deltaMaxMin = max - min;

                for(int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / deltaMaxMin;
                }
            }

            return result;
        }

        private double[,] Normaliztion(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                //Вычисляем среднее значение сигнала
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                //Стандартное квадратичное отклонение нейрона
                var error = 0.0;  
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standardError = Math.Sqrt(error / inputs.GetLength(0));
                
                //Новое значение сигнала
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standardError;
                }
            }
            
            return result;
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            foreach(var neuron in layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for(int j = layers.Count - 2; j >= 0; j--)
            {
                var layer = layers[j];
                var previosLayer = layers[j + 1];

                for(int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for(int k = 0; k < previosLayer.NeuronCount; k++)
                    {
                        var previosNeuron = previosLayer.Neurons[k];
                        var error = previosNeuron.Weights[i] * previosNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return Math.Pow(difference, 2);
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                var layer = layers[i];
                var preveiousLayerSignals = layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(preveiousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for(int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                layers.Add(hiddenLayer);
            }
            
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for(int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            layers.Add(inputLayer);
        }
    }
}

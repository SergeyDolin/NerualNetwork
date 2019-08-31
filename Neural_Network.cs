using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NerualNetwork
{
    public class Nerual_Network
    {
        static void Main(string[] args) // static with a void (or int) return type
        {
        }
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

        public Neuron FeedForward(List<double> inputSignals)
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

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
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
                var neuron = new Neuron(lastLayer.Count, NeuronType.Output);
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
                    var neuron = new Neuron(lastLayer.Count);
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

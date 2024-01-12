using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Test.Neural_Network
{
    public class NeuralNetwork
    {
        #region Fields

        NeuronsLayer[] Layers;

        public double LearningRate;
        public int[] NeuronsMap { get; private set; }
        public int InputsLength => NeuronsMap[0];
        public int OutputsLength => NeuronsMap[NeuronsMap.Length - 1];

        #endregion


        #region Methods

        //      [ Возвращение данных ]
        public double[] GetResult(double[] inputs)
        {
            if (inputs.Length != InputsLength)
                throw new ArgumentException("Length of \"inputs\" elements does not correspond to the number of input neurons");

            double[] results = inputs.Select(i => Neuron.ActivationFunc(i)).ToArray();

            foreach (NeuronsLayer layer in Layers)
                results = layer.GetResult(results);

            return results;
        }

        public double[] GetLastResultOfLayer(int layerIndex = 0) => Layers[Layers.Length - 1 - layerIndex].LastOutputs;

        public NeuralNetwork Clone()
        {
            NeuralNetwork clone = new NeuralNetwork
            {
                LearningRate = LearningRate,
                NeuronsMap = new int[NeuronsMap.Length],
                Layers = Layers.Select(l => l.Clone()).ToArray()
            };
            NeuronsMap.CopyTo(clone.NeuronsMap, 0);

            return clone;
        }


        //      [ Обратное расспостранение ошибки ]
        public void TrainingFromExamples(Dataset dataset, int iterations)
        {
            if (iterations < 0) throw new Exception("\"iterations\" must be greater than zero");


            for (int iter = 0; iter < iterations; iter++)
            {
                for (int i = 0; i < dataset.Length; i++)
                {
                    double[] inputs = dataset[i, DataType.Inputs],
                             corrects = dataset[i, DataType.Corrects];

                    double[] results = GetResult(inputs);


                    double[] errors = new double[results.Length];

                    for (int k = 0; k < corrects.Length; k++)
                        errors[k] = corrects[k] - results[k];

                    
                    for (int k = Layers.Length - 1; k >= 0; k--)
                        errors = Layers[k].Training(errors, LearningRate);
                }
            }
        }

        public void TrainingFromExamples(double[] inputs, double[] corrects, int iterations)
        {
            if (iterations < 0) throw new Exception("\"iterations\" must be greater than zero");

            for (int i = 0; i < iterations; i++)
            {
                double[] results = GetResult(inputs);


                double[] errors = new double[results.Length];

                for (int k = 0; k < errors.Length; k++)
                    errors[k] = corrects[k] - results[k];


                for (int k = Layers.Length - 1; k >= 0; k--)
                    errors = Layers[k].Training(errors, LearningRate);
            }
        }

        //      [ Генетический алгоритм ]
        public NeuralNetwork[] GenerateGeneration(int count, double deviation = 0.1)
        {
            if (count < 1) throw new ArgumentException("\"number\" must be greater than zero");

            if (deviation <= 0) throw new ArgumentException("\"deviations\" must be greater than zero");


            NeuralNetwork[] nets = new NeuralNetwork[count];
            nets[0] = Clone();

            for (int i = 1; i < nets.Length; i++)
            {
                nets[i] = Clone();
                foreach (NeuronsLayer layer in nets[i].Layers)
                    layer.Mutation(deviation);
            }

            return nets;
        }

        public NeuralNetwork PairWith(NeuralNetwork net, double dominance = 0.5)
        {
            // проверка на схожесть двух нейросетей

            bool isCoincidence = true;

            if (NeuronsMap.Length == net.NeuronsMap.Length)
            {
                for (int i = 0; i < NeuronsMap.Length; i++)
                    if (NeuronsMap[i] != net.NeuronsMap[i])
                    {
                        isCoincidence = false;
                        break;
                    }
            }
            else isCoincidence = false;

            if (!isCoincidence) throw new ArgumentException("\"net\" is not a clone of this neural network");

            if (dominance < 0 || dominance > 1) throw new ArgumentException("\"dominance\" must be between 0 and 1");


            NeuralNetwork child = Clone();

            for (int i = 0; i < Layers.Length; i++)
                child.Layers[i] = Layers[i].PairWith(net.Layers[i], dominance);

            return child;
        }


        //      [ Сохранение/считывание нейросети в/из файла ]
        public void SaveToFile(string path, bool overrideFile = false)
        {
            if (File.Exists(path) && !overrideFile) throw new FileLoadException(path);

            List<string> lines = new List<string>
            {
                string.Join(' ', NeuronsMap),
                LearningRate.ToString()
            };


            foreach (NeuronsLayer layer in Layers)
            {
                lines.AddRange(layer.Pack());
                lines.Add("END LAYER");
            }

            File.WriteAllLines(path, lines);
        }

        public void ReadFromFile(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException(path);

            string[] lines = File.ReadAllLines(path);


            NeuronsMap = lines[0].Split(' ').Select(str => int.Parse(str)).ToArray();
            LearningRate = double.Parse(lines[1]);
            BuildNet(NeuronsMap);


            List<string> pack = new List<string>();
            int layerIdx = 0;

            for (int i = 2; i < lines.Length; i++)
            {
                if (lines[i] != "END LAYER")
                    pack.Add(lines[i]);
                else
                {
                    Layers[layerIdx++].Unpack(pack);
                    pack.Clear();
                }
            }
        }


        //      [ Инициализация нейросети ]
        public void BuildNet(int[] neuronsMap)
        {
            if (neuronsMap.Length < 2) throw new ArgumentException("\"neuronsMap\" must contain at least 2 elements");

            foreach (int neuronsNumber in neuronsMap)
                if (neuronsNumber < 1) throw new ArgumentException("Each \"neuronsMap\" element must be greater than zero");

            NeuronsMap = new int[neuronsMap.Length];
            neuronsMap.CopyTo(NeuronsMap, 0);


            // neuronsMap определяет количество слоёв и количество нейронов в каждом слое:
            //
            //                входы   слой 1   слой 2   слой 3
            // neuronsMap = {   6   ,   4    ,   6    ,   3    };

            Layers = new NeuronsLayer[neuronsMap.Length - 1];

            for (int i = 0; i < Layers.Length; i++)
                Layers[i] = new NeuronsLayer(neuronsMap[i + 1], neuronsMap[i]);
        }

        #endregion


        public NeuralNetwork(int[] neuronsMap, double learningRate = 0.01)
        {
            LearningRate = learningRate;
            BuildNet(neuronsMap);
        }

        public NeuralNetwork(string path) => ReadFromFile(path);

        NeuralNetwork() { }
    }
}

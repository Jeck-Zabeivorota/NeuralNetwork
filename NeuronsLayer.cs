using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Test.Neural_Network
{
    class NeuronsLayer
    {
        #region Fields

        Neuron[] Neurons;

        public readonly double[] LastInputs, LastOutputs;

        public int NeuronsLength => Neurons.Length;

        #endregion


        #region Methods

        //      [ Возвращение данных ]
        public double[] GetResult(double[] inputs)
        {
            double[] outputs = Neurons.Select(n => n.GetResult(inputs)).ToArray();

            inputs.CopyTo(LastInputs, 0);
            outputs.CopyTo(LastOutputs, 0);

            return outputs;
        }

        public NeuronsLayer Clone() => new NeuronsLayer { Neurons = Neurons.Select(n => n.Clone()).ToArray() };


        //      [ Обратное расспостранение ошибки ]
        public double[] Training(double[] errors, double learningRate)
        {
            // создание массива для ошибок предыдущего слоя и заполнение его нулями
            // [ количество весов любого нейрона
            //   соответствует количеству нейронов в предыдущем слое ]
            double[] inputsErrors = new double[LastInputs.Length];

            for (int i = 0; i < inputsErrors.Length; i++) inputsErrors[i] = 0;

            // тренировка нейронов
            for (int i = 0; i < Neurons.Length; i++)
            {
                // получение и сложение всех "мнений" от нейронов,
                // по поводу степени ошибки каждого нейрона из предыдущего слоя:
                //
                // мнение нейрона 1  ->  вх. ошибка 1 = 3,  вх. ошибка 2 = 8
                // мнение нейрона 2  ->  вх. ошибка 1 = 1,  вх. ошибка 2 = 4
                // среднее мнений    ->  вх. ошибка 1 = 2,  вх. ошибка 2 = 6

                double[] inputsErrorsFromNeuron = Neurons[i].Training(errors[i], LastInputs, LastOutputs[i], learningRate);

                for (int k = 0; k < inputsErrors.Length; k++)
                    inputsErrors[k] += inputsErrorsFromNeuron[k];
            }
            // усреднение входных ошибок
            for (int i = 0; i < inputsErrors.Length; i++)
                inputsErrors[i] /= Neurons.Length;

            return inputsErrors;
        }


        //      [ Генетический алгоритм ]
        public void Mutation(double deviation)
        {
            foreach (Neuron neuron in Neurons)
                neuron.Mutation(deviation);
        }

        public NeuronsLayer PairWith(NeuronsLayer layer, double dominance)
        {
            NeuronsLayer child = Clone();

            for (int i = 0; i < Neurons.Length; i++)
                child.Neurons[i] = Neurons[i].PairWith(layer.Neurons[i], dominance);

            return child;
        }


        //      [ Упаковка/розпаковка даных слоя для записи в файл ]
        public string[] Pack() => Neurons.Select(n => n.Pack()).ToArray();

        public void Unpack(List<string> pack)
        {
            for (int i = 0; i < Neurons.Length; i++)
                Neurons[i].Unpack(pack[i]);
        }

        #endregion


        public NeuronsLayer(int neuronsNumber, int neuronMemoryLength, int inputsNumber)
        {
            LastInputs = new double[inputsNumber];
            LastOutputs = new double[neuronsNumber];

            Neurons = new Neuron[neuronsNumber];

            for (int i = 0; i < neuronsNumber; i++)
                Neurons[i] = new Neuron(inputsNumber, neuronMemoryLength);
        }

        NeuronsLayer() { }
    }
}

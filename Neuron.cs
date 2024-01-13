using System;
using System.Collections.Generic;
using System.Linq;

namespace Test.Neural_Network
{
    class Neuron
    {
        #region Fields

        readonly List<double> LastOutputs = new List<double>();

        double[] InputWeights, OutputWeights;
        double Bias;

        #endregion


        #region Methods

        public void ClearMemory() => LastOutputs.Clear();


        //      [ Возвращение данных ]
        public double GetResult(double[] inputs)
        {
            double sum = 0;

            for (int i = 0; i < inputs.Length; i++)
                sum += inputs[i] * InputWeights[i];

            for (int i = 0; i < LastOutputs.Count; i++)
                sum += LastOutputs[i] * OutputWeights[i];

            sum += Bias;
            double result = ActivationFunc(sum);

            if (OutputWeights.Length > 0)
            {
                LastOutputs.Insert(0, result);
                if (LastOutputs.Count > OutputWeights.Length) LastOutputs.RemoveAt(LastOutputs.Count - 1);
            }

            return result;
        }

        public Neuron Clone()
        {
            Neuron clone = new Neuron
            {
                InputWeights = new double[InputWeights.Length],
                OutputWeights = new double[OutputWeights.Length],
                Bias = Bias
            };
            InputWeights.CopyTo(clone.InputWeights, 0);
            OutputWeights.CopyTo(clone.OutputWeights, 0);

            return clone;
        }


        //      [ Обратное расспостранение ошибки ]
        public static double ActivationFunc(double value) => 1 / (1 + Math.Exp(-value));

        double Derivative(double value) => value * (1 - value);

        public double[] Training(double error, double[] lastInputs, double lastOutput, double learningRate)
        {
            double[] inputsErrors = new double[InputWeights.Length];

            // [ чем ближе выходное значение нейрона к 0 или 1 (в сигмоиде),
            //   тем меньше его подвижность (как железный шарик между магнитами) ]
            // увеличивать или уменьшать веса зависит от знака ошибки(error): - меньше, + больше
            //
            // величина изменения = +-величина ошибки * изменяемость * сглаживание
            double gradient = error * Derivative(lastOutput) * learningRate;

            for (int i = 0; i < InputWeights.Length; i++)
            {
                // распостранение ошибки на предыдущий слой с учётом влияния веса
                inputsErrors[i] = error * InputWeights[i];

                // корректировка весов с учётом влияния входного значение
                InputWeights[i] += gradient * lastInputs[i];
            }

            for (int i = 0; i < LastOutputs.Count; i++)
                OutputWeights[i] += gradient * LastOutputs[i];

            // корректировка смещения
            Bias += gradient;

            return inputsErrors;
        }


        //      [ Генетический алгоритм ]
        double Deviate(double deviation)
        {
            Random rand = new Random((int)DateTime.Now.Ticks);

            // определение размера отклонения
            double result = deviation * rand.NextDouble();

            // определение в большую, либо в меньшую сторону будет отклонение
            return rand.Next(2) == 0 ? result : -result;
        }

        public void Mutation(double deviation)
        {
            for (int i = 0; i < InputWeights.Length; i++)
                InputWeights[i] += Deviate(deviation);

            for (int i = 0; i < OutputWeights.Length; i++)
                OutputWeights[i] += Deviate(deviation);

            Bias += Deviate(deviation);
        }

        public Neuron PairWith(Neuron neuron, double dominance)
        {
            Random rand = new Random((int)DateTime.Now.Ticks);
            Neuron child = Clone();

            // замена указаного процента весов сторонего нейрона на веса даного нейрона
            for (int i = 0; i < InputWeights.Length; i++)
                if (rand.NextDouble() < dominance)
                    child.InputWeights[i] = neuron.InputWeights[i];

            for (int i = 0; i < OutputWeights.Length; i++)
                if (rand.NextDouble() < dominance)
                    child.OutputWeights[i] = neuron.OutputWeights[i];

            if (rand.NextDouble() < dominance)
                child.Bias = neuron.Bias;

            return child;
        }


        //      [ Упаковка/разпаковка даных нейрона для записи в файл ]
        public string Pack()
        {
            string pack = $"{Bias} {string.Join(" ", InputWeights)}";

            if (OutputWeights.Length > 0)
                pack += $"|{string.Join(" ", OutputWeights)}";

            return pack;
        }

        double[] StringToDoubleArray(string line) => line.Split(' ').Select(str => double.Parse(str)).ToArray();

        public void Unpack(string pack)
        {
            int firstSeparIdx = pack.IndexOf(' ');
            Bias = double.Parse(pack.Substring(0, firstSeparIdx));

            int secondSeparIdx = pack.LastIndexOf('|');

            if (secondSeparIdx == -1)
            {
                InputWeights = StringToDoubleArray(pack.Substring(firstSeparIdx + 1)); // 01-234-56
                OutputWeights = new double[0];
            }
            else
            {
                InputWeights = StringToDoubleArray(pack.Substring(firstSeparIdx + 1, secondSeparIdx - firstSeparIdx - 1));
                OutputWeights = StringToDoubleArray(pack.Substring(secondSeparIdx + 1));
            }
        }

        #endregion


        public Neuron(int inputsNumber, int memoryLength)
        {
            InputWeights = new double[inputsNumber];
            OutputWeights = new double[memoryLength];

            Random rand = new Random((int)DateTime.Now.Ticks);

            for (int i = 0; i < inputsNumber; i++)
                InputWeights[i] = rand.NextDouble();

            for (int i = 0; i < memoryLength; i++)
                OutputWeights[i] = rand.NextDouble();

            Bias = rand.NextDouble();
        }

        Neuron() { }
    }
}

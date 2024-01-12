using System;
using System.Linq;

namespace Test.Neural_Network
{
    class Neuron
    {
        #region Fields

        double[] Weights;
        double Bias;

        public int WeightsLength => Weights.Length;

        #endregion


        #region Methods

        //      [ Возвращение данных ]
        public double GetResult(double[] inputs)
        {
            double sum = 0;
            
            for (int i = 0; i < inputs.Length; i++)
                sum += inputs[i] * Weights[i];
            
            sum += Bias;

            return ActivationFunc(sum);
        }

        public Neuron Clone()
        {
            Neuron clone = new Neuron
            {
                Weights = new double[Weights.Length],
                Bias = Bias
            };
            Weights.CopyTo(clone.Weights, 0);

            return clone;
        }


        //      [ Обратное расспостранение ошибки ]
        public static double ActivationFunc(double value) => 1 / (1 + Math.Exp(-value));

        double Derivative(double value) => value * (1 - value);

        public double[] Training(double error, double[] lastInputs, double lastOutput, double learningRate)
        {
            double[] inputsErrors = new double[Weights.Length];

            // [ чем ближе выходное значение нейрона к 0 или 1 (в сигмоиде),
            //   тем меньше его подвижность (как железный шарик между магнитами) ]
            // увеличивать или уменьшать веса зависит от знака ошибки(error): - меньше, + больше
            //
            // величина изменения = +-величина ошибки * изменяемость * сглаживание
            double gradient = error * Derivative(lastOutput) * learningRate;

            for (int i = 0; i < Weights.Length; i++)
            {
                // распостранение ошибки на предыдущий слой с учётом влияния веса
                inputsErrors[i] = error * Weights[i];

                // корректировка весов с учётом влияния входного значение
                Weights[i] += gradient * lastInputs[i];
            }

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
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] += Deviate(deviation);

            Bias += Deviate(deviation);
        }

        public Neuron PairWith(Neuron neuron, double dominance)
        {
            Random rand = new Random((int)DateTime.Now.Ticks);
            Neuron child = Clone();

            // замена указаного процента весов сторонего нейрона на веса даного нейрона
            for (int i = 0; i < Weights.Length; i++)
                if (rand.NextDouble() < dominance)
                    child.Weights[i] = neuron.Weights[i];

            if (rand.NextDouble() < dominance)
                child.Bias = neuron.Bias;

            return child;
        }


        //      [ Упаковка/разпаковка даных нейрона для записи в файл ]
        public string Pack() => $"{string.Join(' ', Weights)} {Bias}";

        public void Unpack(string pack)
        {
            int biasStartIdx = pack.LastIndexOf(' ') + 1;
            Bias = double.Parse(pack.Substring(biasStartIdx));
            pack = pack.Remove(biasStartIdx - 1);

            Weights = pack.Split(' ').Select(str => double.Parse(str)).ToArray();
        }

        #endregion


        public Neuron(int inputsLength)
        {
            Weights = new double[inputsLength];

            Random rand = new Random((int)DateTime.Now.Ticks);

            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = rand.NextDouble();

            Bias = rand.NextDouble();
        }

        Neuron() { }
    }
}

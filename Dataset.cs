using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Test.Neural_Network
{
    public enum DataType { Inputs, Corrects }

    public class Dataset
    {
        #region Fields

        readonly List<double[]> Inputs = new List<double[]>();
        readonly List<double[]> Corrects = new List<double[]>();


        public double[] this[int index, DataType type]
        {
            get
            {
                if (type == DataType.Inputs)
                    return Inputs[index]; // входние сигналы
                else
                    return Corrects[index]; // верные выходние сигналы
            }
            set
            {
                if (type == DataType.Inputs)
                    Inputs[index] = value; // заменить входние сигналы
                else
                    Corrects[index] = value; // заменить верные выходние сигналы
            }
        }

        public int Length => Inputs.Count;

        #endregion


        #region Methods

        public void Add(double[] inputs, double[] corrects)
        {
            Inputs.Add(inputs);
            Corrects.Add(corrects);
        }

        public void RemoveAt(int index)
        {
            Inputs.RemoveAt(index);
            Corrects.RemoveAt(index);
        }


        double[] StringToDoubleArray(string line) => line.Split(' ').Select(str => double.Parse(str)).ToArray();

        public void SaveToFile(string path, bool overrideFile = false)
        {
            if (File.Exists(path) && !overrideFile) throw new FileLoadException(path);

            string[] lines = new string[Inputs.Count + Corrects.Count];

            int dataIdx = 0;
            for (int i = 0; i < lines.Length; i += 2)
            {
                lines[i] = string.Join(" ", Inputs[dataIdx]);
                lines[i + 1] = string.Join(" ", Corrects[dataIdx++]);
            }

            File.WriteAllLines(path, lines);
        }

        public void ReadFromFile(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException(path);

            string[] lines = File.ReadAllLines(path);

            Inputs.Clear();
            Corrects.Clear();

            for (int i = 0; i < lines.Length; i += 2)
            {
                Inputs.Add(StringToDoubleArray(lines[i]));
                Corrects.Add(StringToDoubleArray(lines[i + 1]));
            }
        }

        #endregion


        public Dataset() { }

        public Dataset(string path) => ReadFromFile(path);
    }
}

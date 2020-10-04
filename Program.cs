using System.IO.Compression;
using System.Net;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UMAP;
using XPlot.Plotly;

namespace UMAPExampleInCSharp
{
    class Program
    {
        // URLs to download MNIST data
        const string LABEL_FILE_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
        const string IMAGE_FILE_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";

        // Name of MNIST files
        const string LABEL_FILE_NAME = "t10k-labels-idx1-ubyte.gz";
        const string IMAGE_FILE_NAME = "t10k-images-idx3-ubyte.gz";

        // Main
        static void Main(string[] args)
        {
            // Download the MNIST data from Mr. YannAndréLeCun's web site
            using (var wc = new WebClient())
            {
                if (!File.Exists(LABEL_FILE_NAME))
                    wc.DownloadFile(LABEL_FILE_URL, LABEL_FILE_NAME);
                if (!File.Exists(IMAGE_FILE_NAME))
                    wc.DownloadFile(IMAGE_FILE_URL, IMAGE_FILE_NAME);
            }

            // Load the MNIST labels
            var labels = GetLabels().ToArray();

            // Load the MNIST image data
            var pixels = GetImages().ToArray();

            // Dimension reduction with UMAP
            var umap = new Umap();
            var epochs = umap.InitializeFit(pixels);
            for (var i = 0; i < epochs; ++i)
                umap.Step();
            var embedding = umap.GetEmbedding().AsEnumerable();

            // Convert the embedding into chart data
            var graph = new Graph.Scatter()
            {
                x = embedding.Select((o) => o[0]),
                y = embedding.Select((o) => o[1]),
                text = labels.Select((o) => o.ToString()),
                mode = "markers",
                marker = new Graph.Marker { color = labels, colorscale = "Rainbow", showscale = true },
            };

            var chart = Chart.Plot(graph);
            chart.WithTitle("MNIST Embedded via UMAP");
            chart.WithXTitle("X");
            chart.WithYTitle("Y");
            chart.WithSize(800, 800);
            chart.Show();
        }

        // Extract gz file into a byte array
        private static byte[] Unzip(string filePath)
        {
            using (var fs = new FileStream(filePath, FileMode.Open))
            {
                using (var gzs = new GZipStream(fs, CompressionMode.Decompress))
                {
                    using (var ms = new MemoryStream())
                    {
                        gzs.CopyTo(ms);
                        return ms.ToArray();
                    }
                }
            }
        }

        // Get labels of MNIST from file
        private static IEnumerable<float> GetLabels()
        {
            var binData = Unzip(LABEL_FILE_NAME);
            // Id of MNIST data(2049)
            var magicNumber = BitConverter.ToInt32(binData.Take(4).Reverse().ToArray());
            // Number of images(10000)
            var numOfItems = BitConverter.ToInt32(binData.Skip(4).Take(4).Reverse().ToArray());

            return binData.Skip(8).Select((o) => (float)o);
        }

        // Get data of MNIST from file
        private static IEnumerable<float[]> GetImages()
        {
            var binData = Unzip(IMAGE_FILE_NAME);
            // Magic number(2051)
            var magicNumber = BitConverter.ToInt32(binData.Take(4).Reverse().ToArray());
            // Number of images(10000)
            var numOfItems = BitConverter.ToInt32(binData.Skip(4).Take(4).Reverse().ToArray());
            // Number of rows(28)
            var numOfRows = BitConverter.ToInt32(binData.Skip(8).Take(4).Reverse().ToArray());
            // Number of columns(28)
            var numOfColumns = BitConverter.ToInt32(binData.Skip(12).Take(4).Reverse().ToArray());

            var length = numOfRows * numOfColumns;

            for (int i = 16, max = binData.Count(); i < max; i += length)
            {
                // Byte array into chunks
                var chunk = binData.Skip(i).Select((o) => (float)o).Take(length).ToArray();
                yield return chunk;
            }
        }
    }
}
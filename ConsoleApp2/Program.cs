using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

HouseData[] list =
{
    new HouseData(){Size = 1.3F, Price = 2.1F},
    new HouseData(){Size = 1.6F, Price = 2.4F},
    new HouseData(){Size = 1.9F, Price = 2.8F},
    new HouseData(){Size = 2.3F, Price = 3.2F},
    new HouseData(){Size = 2.6F, Price = 4.1F}
};

var mlContext = new MLContext();
IDataView dt = mlContext.Data.LoadFromEnumerable(list);
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" }).Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 1000));
var model = pipeline.Fit(dt);
var size = new HouseData() { Size = 2.7F };
var predictionScore = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model);
var score =predictionScore.Predict(size);
var metrics = mlContext.Regression.Evaluate(dt, "Size", "Price");
Console.WriteLine($"{ score.Price * 100}K score : metrics.RSquared * 0.100");

//var context = new MLContext(); Auto ML
//IDataView dtt = mlContext.Data.LoadFromEnumerable<HouseData>(list);
//var exp = context.Auto().CreateRegressionExperiment(maxExperimentTimeInSeconds: 30);
//var results = exp.Execute(trainData : dtt, labelColumnName: nameof(HouseData.Price));
//Console.WriteLine($"Algoritma:  {results.BestRun.TrainerName}");
//Console.WriteLine($"R^2:  {results.BestRun.ValidationMetrics.RSquared}");

public class HouseData
{
    public float Size { get; set; }
    public float Price { get; set; }
}
public class Prediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}
